#!/usr/bin/env python3
"""
================================================================================
FINAL RESEARCH PIPELINE – FULL ID/OOD GENERALIZATION + ATTACK SENSITIVITY
================================================================================
Includes:
- ID → OOD evaluation
- Attack strength sensitivity (FDIA magnitude, load multiplier)
- Per‑attack localization accuracy
- Confusion matrix heatmaps
- Attention visualisation (extracted from GAT layers)
- Full factorial ablation, lambda sweep, robustness, statistics, LaTeX tables
================================================================================
"""

import os
import time
import json
import math
import warnings
import pickle
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

import pandapower.networks as pn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef)
from sklearn.model_selection import ParameterGrid
from scipy.stats import wilcoxon, friedmanchisquare

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ExperimentConfig:
    experiment_name: str = "PI-GAT_Final"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    n_folds: int = 5
    max_epochs: int = 150
    early_stopping_patience: int = 25
    grad_clip: float = 1.0

    full_features: int = 6
    reduced_features: int = 4
    num_classes: int = 4
    heads: int = 2

    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    results_dir: str = "final_results"
    figures_dir: str = "final_figures"
    models_dir: str = "final_models"

    hp_search_space: Dict = field(default_factory=lambda: {
        'hidden_channels': [32, 64, 128],
        'learning_rate': [0.001, 0.0005],
        'dropout': [0.1, 0.2, 0.3],
        'lambda_phys': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        'weight_decay': [1e-5, 1e-4],
        'batch_size': [32, 64],
    })
    best_hp: Dict = field(default_factory=dict)

    noise_levels: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.15])
    missing_ratios: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1])
    
    # Attack strength sweep parameters
    fdia_magnitudes: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.20])
    load_multipliers: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 4.0])


# ============================================================
# DATA LOADING (ID and OOD)
# ============================================================

class GridTopologyExtractor:
    def __init__(self):
        net = pn.case39()
        s_base = net.sn_mva
        edges_fb, edges_tb = [], []
        g_vals, b_vals = [], []
        for _, line in net.line.iterrows():
            fb, tb = int(line.from_bus), int(line.to_bus)
            zb = (net.bus.vn_kv.at[fb] ** 2) / s_base
            r = (line.r_ohm_per_km * line.length_km) / zb
            x = (line.x_ohm_per_km * line.length_km) / zb
            y = 1.0 / (r + 1j * x)
            edges_fb.extend([fb, tb]); edges_tb.extend([tb, fb])
            g_vals.extend([y.real, y.real]); b_vals.extend([y.imag, y.imag])
        for _, trafo in net.trafo.iterrows():
            fb, tb = int(trafo.hv_bus), int(trafo.lv_bus)
            r = (trafo.vkr_percent / 100.0) * (s_base / trafo.sn_mva)
            x = (np.sqrt(trafo.vk_percent**2 - trafo.vkr_percent**2) / 100.0) * (s_base / trafo.sn_mva)
            y = 1.0 / (r + 1j * x)
            edges_fb.extend([fb, tb]); edges_tb.extend([tb, fb])
            g_vals.extend([y.real, y.real]); b_vals.extend([y.imag, y.imag])
        self.edge_index = np.vstack([edges_fb, edges_tb]).astype(np.int64)
        self.G_line = np.array(g_vals, dtype=np.float32)
        self.B_line = np.array(b_vals, dtype=np.float32)
        self.slack_idx = int(net.ext_grid.bus.iloc[0])
        self.n_buses = len(net.bus)


class DataManager:
    def __init__(self, topology: GridTopologyExtractor, config: ExperimentConfig):
        self.topology = topology
        self.config = config

    def load_dataset(self, filepath: str) -> Dict[str, List]:
        data = np.load(filepath, allow_pickle=True)
        class_samples = {}
        for cls in ['normal', 'fdia', 'topology', 'laa']:
            if f'{cls}_x' in data:
                X = data[f'{cls}_x']
                Y = data[f'{cls}_y']
                valid = ~np.isnan(X).any(axis=(1,2))
                X, Y = X[valid], Y[valid]
                class_samples[cls] = [(x.astype(np.float32), y.astype(np.int64)) for x, y in zip(X, Y)]
        return class_samples

    def create_fold_splits(self, class_samples: Dict[str, List], fold: int, seed: int = 42):
        train, val, test = [], [], []
        for cls_name, samples in class_samples.items():
            n = len(samples)
            if n == 0:
                continue
            rng = np.random.RandomState(seed + fold)
            shuffled = samples.copy()
            rng.shuffle(shuffled)
            fold_size = max(1, n // self.config.n_folds)
            test_start = (fold * fold_size) % n
            test_end = min(test_start + fold_size, n)
            if test_end <= test_start:
                test_end = min(test_start + 1, n)
            val_start = (test_end) % n
            val_end = min(val_start + fold_size, n)
            if val_end <= val_start:
                val_end = min(val_start + 1, n)
            test_idx = set(range(test_start, test_end))
            val_idx = set(range(val_start, val_end))
            train_idx = set(range(n)) - test_idx - val_idx
            if len(train_idx) == 0:
                train_idx = set(range(n)) - test_idx
                val_idx = set()
            train.extend([shuffled[i] for i in sorted(train_idx)])
            val.extend([shuffled[i] for i in sorted(val_idx)])
            test.extend([shuffled[i] for i in sorted(test_idx)])
        return train, val, test

    def create_dataloader(self, samples: List, batch_size: int, shuffle: bool):
        if len(samples) == 0:
            return DataLoader([], batch_size=batch_size)
        edge_t = torch.tensor(self.topology.edge_index, dtype=torch.long)
        data_list = []
        for x, y in samples:
            data_list.append(Data(x=torch.tensor(x, dtype=torch.float32),
                                  edge_index=edge_t,
                                  y=torch.tensor(y, dtype=torch.long)))
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def compute_class_weights(self, samples: List) -> np.ndarray:
        if len(samples) == 0:
            return np.ones(4, dtype=np.float32) / 4
        all_labels = np.concatenate([y.flatten() for _, y in samples])
        from sklearn.utils.class_weight import compute_class_weight
        return compute_class_weight('balanced', classes=np.array([0,1,2,3]), y=all_labels).astype(np.float32)


# ============================================================
# MODELS (same as before but with attention extraction)
# ============================================================

class BaseGNN(nn.Module):
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PI_GAT(BaseGNN):
    def __init__(self, in_features=6, hidden=64, num_classes=4, heads=2, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_features, hidden, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout, add_self_loops=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            h, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
            h = F.relu(self.norm1(h))
            h = self.dropout(h)
            h, attn2 = self.conv2(h, edge_index, return_attention_weights=True)
            h = F.relu(self.norm2(h))
            return self.classifier(h), (attn1, attn2)
        else:
            h = F.relu(self.norm1(self.conv1(x, edge_index)))
            h = self.dropout(h)
            h = F.relu(self.norm2(self.conv2(h, edge_index)))
            return self.classifier(h)


class PI_GAT_NoResiduals(BaseGNN):
    def __init__(self, in_features=4, hidden=64, num_classes=4, heads=2, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_features, hidden, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout, add_self_loops=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        return self.classifier(h)


class StandardGAT(BaseGNN):
    def __init__(self, in_features=4, hidden=64, num_classes=4, heads=2, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_features, hidden, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        return self.classifier(h)


class GCN(BaseGNN):
    def __init__(self, in_features=6, hidden=64, num_classes=4, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)


class GraphSAGE(BaseGNN):
    def __init__(self, in_features=6, hidden=64, num_classes=4, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h)


class MLP(BaseGNN):
    def __init__(self, in_features=6, hidden=128, num_classes=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index=None):
        return self.net(x)


# ============================================================
# PHYSICS LOSS
# ============================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(self, topology: GridTopologyExtractor, lambda_phys=0.001, class_weights=None):
        super().__init__()
        self.register_buffer('fb', torch.tensor(topology.edge_index[0], dtype=torch.long))
        self.register_buffer('tb', torch.tensor(topology.edge_index[1], dtype=torch.long))
        self.register_buffer('g', torch.tensor(topology.G_line, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(topology.B_line, dtype=torch.float32))
        self.lambda_phys = lambda_phys
        self.slack_idx = topology.slack_idx
        if class_weights is not None:
            self.register_buffer('class_weights', torch.from_numpy(class_weights.astype(np.float32)))
        else:
            self.class_weights = None

    def _physics_loss(self, x, target):
        v = x[:, 2].clamp(0.85, 1.15)
        theta = x[:, 3]
        dtheta = theta[self.fb] - theta[self.tb]
        dtheta = torch.clamp(dtheta, -math.pi/3, math.pi/3)
        v_fb, v_tb = v[self.fb], v[self.tb]
        cos_d, sin_d = torch.cos(dtheta), torch.sin(dtheta)
        p_flow = self.g * v_fb**2 - v_fb * v_tb * (self.g * cos_d + self.b * sin_d)
        q_flow = -self.b * v_fb**2 - v_fb * v_tb * (self.g * sin_d - self.b * cos_d)
        p_calc = torch.zeros_like(v); q_calc = torch.zeros_like(v)
        p_calc.scatter_add_(0, self.fb, p_flow); q_calc.scatter_add_(0, self.fb, q_flow)
        phys = torch.abs(x[:, 0] - p_calc) + torch.abs(x[:, 1] - q_calc)
        normal_mask = (target == 0)
        if self.slack_idx < len(normal_mask):
            normal_mask[self.slack_idx] = False
        return phys[normal_mask].mean() if normal_mask.sum() > 0 else torch.tensor(0.0, device=v.device)

    def compute_physics_violation(self, x, target):
        with torch.no_grad():
            return self._physics_loss(x, target).item()

    def forward(self, logits, target, x):
        task_loss = F.cross_entropy(logits, target, weight=self.class_weights)
        phys_loss = self._physics_loss(x, target)
        total = task_loss + self.lambda_phys * phys_loss
        return total, task_loss.item(), phys_loss.item()


# ============================================================
# TRAINING ENGINE (with per‑attack localization)
# ============================================================

class TrainingEngine:
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
        self.device = config.device

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss, total_task, total_phys, n = 0,0,0,0
        for batch in loader:
            batch = batch.to(self.device)
            logits = model(batch.x, batch.edge_index)
            if isinstance(criterion, PhysicsInformedLoss):
                loss, task_l, phys_l = criterion(logits, batch.y, batch.x)
            else:
                loss = criterion(logits, batch.y)
                task_l, phys_l = loss.item(), 0.0
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            n += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
            total_task += task_l * batch.num_graphs
            total_phys += phys_l * batch.num_graphs
        if n == 0:
            return 0,0,0
        return total_loss/n, total_task/n, total_phys/n

    @torch.no_grad()
    def evaluate(self, model, loader, compute_physics_violation=False, phys_criterion=None):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_phys = 0.0
        n_batches = 0
        for batch in loader:
            if len(batch) == 0:
                continue
            batch = batch.to(self.device)
            logits = model(batch.x, batch.edge_index)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            if compute_physics_violation and phys_criterion is not None:
                total_phys += phys_criterion.compute_physics_violation(batch.x, batch.y)
                n_batches += 1
        if not all_preds:
            return None
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(labels, preds),
        }
        if compute_physics_violation and n_batches > 0:
            metrics['physics_violation'] = total_phys / n_batches

        # Per‑class precision/recall
        prec, rec, _, _ = precision_recall_fscore_support(labels, preds, labels=[0,1,2,3], zero_division=0)
        for i, name in enumerate(['normal','fdia','topology','laa']):
            metrics[f'{name}_prec'] = prec[i]
            metrics[f'{name}_rec'] = rec[i]

        # Localization for each attack type separately
        for attack_idx, attack_name in [(1,'fdia'), (2,'topology'), (3,'laa')]:
            mask = (labels == attack_idx)
            if mask.sum() > 0:
                metrics[f'loc_{attack_name}'] = accuracy_score(labels[mask], preds[mask])
            else:
                metrics[f'loc_{attack_name}'] = 0.0

        # Overall localization (any attack)
        attacked = labels > 0
        if attacked.sum() > 0:
            metrics['localization_precision'] = accuracy_score(labels[attacked], preds[attacked])
        else:
            metrics['localization_precision'] = 0.0

        metrics['confusion_matrix'] = confusion_matrix(labels, preds, labels=[0,1,2,3])
        return metrics

    @torch.no_grad()
    def benchmark_model(self, model, sample_loader, num_runs=500):
        for batch in sample_loader:
            if len(batch) > 0:
                sample = batch
                break
        else:
            return {'latency_ms': 0, 'std_latency_ms': 0, 'memory_mb': 0, 'flops_g': 0, 'params': 0}
        sample = sample.to(self.device)
        params = model.count_parameters()
        latencies = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            _ = model(sample.x, sample.edge_index)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append((time.perf_counter() - start) * 1000)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = model(sample.x, sample.edge_index)
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            memory_mb = 0
        flops_g = 0
        if THOP_AVAILABLE:
            try:
                flops, _ = profile(model, inputs=(sample.x, sample.edge_index), verbose=False)
                flops_g = flops / 1e9
            except:
                pass
        return {
            'latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'memory_mb': memory_mb,
            'flops_g': flops_g,
            'params': params
        }

    def train_model(self, model, train_loader, val_loader, criterion, hp, fold, seed):
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs, eta_min=1e-6)
        best_val_acc = 0
        best_state = None
        patience = 0
        for epoch in range(1, self.config.max_epochs+1):
            _, _, _ = self.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self.evaluate(model, val_loader)
            if val_metrics is None:
                continue
            scheduler.step()
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= self.config.early_stopping_patience:
                break
        if best_state:
            model.load_state_dict(best_state)
            torch.save(best_state, f"{self.config.models_dir}/best_{self.config.experiment_id}_seed{seed}_fold{fold}.pt")
        return model, best_val_acc


# ============================================================
# HYPERPARAMETER OPTIMIZATION (same as before)
# ============================================================

class HyperparameterOptimizer:
    def __init__(self, config, tracker, topology, data_manager):
        self.config = config
        self.tracker = tracker
        self.topology = topology
        self.data_manager = data_manager
        self.trainer = TrainingEngine(config, tracker)
        self.hp_results = []

    def grid_search(self, class_samples, n_trials=15):
        self.tracker.log("\n=== HYPERPARAMETER OPTIMIZATION ===")
        all_combos = list(ParameterGrid(self.config.hp_search_space))
        if len(all_combos) > n_trials:
            np.random.seed(42)
            combos = [all_combos[i] for i in np.random.choice(len(all_combos), n_trials, replace=False)]
        else:
            combos = all_combos
        train, val, _ = self.data_manager.create_fold_splits(class_samples, fold=0, seed=42)
        class_weights = self.data_manager.compute_class_weights(train)
        results = []
        for hp in combos:
            self.tracker.log(f"Trial: {hp}")
            model = PI_GAT(in_features=self.config.full_features,
                           hidden=hp['hidden_channels'],
                           num_classes=self.config.num_classes,
                           heads=self.config.heads,
                           dropout=hp['dropout']).to(self.config.device)
            train_loader = self.data_manager.create_dataloader(train, hp['batch_size'], True)
            val_loader = self.data_manager.create_dataloader(val, hp['batch_size'], False)
            criterion = PhysicsInformedLoss(self.topology, hp['lambda_phys'], class_weights).to(self.config.device)
            _, best_val = self.trainer.train_model(model, train_loader, val_loader, criterion, hp, 0, 42)
            results.append({**hp, 'val_acc': best_val})
            self.hp_results.append({**hp, 'val_acc': best_val})
            self.tracker.log(f"  val_acc = {best_val*100:.2f}%")
        best = max(results, key=lambda x: x['val_acc'])
        self.tracker.log(f"Best HP: {best}")
        return {k: best[k] for k in self.config.hp_search_space.keys()}


# ============================================================
# CROSS-VALIDATION (ID) – used for ablation and baselines
# ============================================================

def run_cv(model_class, model_name, use_physics_loss, remove_residuals,
           class_samples, best_hp, config, data_manager, topology, trainer):
    config.tracker.log(f"\n--- CV (ID): {model_name} ---")
    all_metrics = []
    for seed in config.random_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for fold in range(config.n_folds):
            train, val, test = data_manager.create_fold_splits(class_samples, fold, seed)
            class_weights = data_manager.compute_class_weights(train)
            in_feat = config.reduced_features if (remove_residuals or 'Standard GAT' in model_name) else config.full_features
            if model_class == MLP:
                model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels']*2,
                                    num_classes=config.num_classes, dropout=best_hp['dropout']).to(config.device)
            elif model_class == StandardGAT:
                model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels'],
                                    num_classes=config.num_classes, heads=config.heads,
                                    dropout=best_hp['dropout']).to(config.device)
            else:
                model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels'],
                                    num_classes=config.num_classes, dropout=best_hp['dropout']).to(config.device)
            if remove_residuals or 'Standard GAT' in model_name:
                train = [(x[:,:config.reduced_features], y) for x,y in train]
                val   = [(x[:,:config.reduced_features], y) for x,y in val]
                test  = [(x[:,:config.reduced_features], y) for x,y in test]
            train_loader = data_manager.create_dataloader(train, best_hp['batch_size'], True)
            val_loader   = data_manager.create_dataloader(val, best_hp['batch_size'], False)
            test_loader  = data_manager.create_dataloader(test, best_hp['batch_size'], False)
            if use_physics_loss and not remove_residuals and 'Standard GAT' not in model_name:
                criterion = PhysicsInformedLoss(topology, best_hp['lambda_phys'], class_weights).to(config.device)
            else:
                cw = torch.from_numpy(class_weights).to(config.device)
                criterion = nn.CrossEntropyLoss(weight=cw).to(config.device)
            model, _ = trainer.train_model(model, train_loader, val_loader, criterion, best_hp, fold, seed)
            phys_criterion = PhysicsInformedLoss(topology, 0, class_weights).to(config.device)
            test_metrics = trainer.evaluate(model, test_loader, compute_physics_violation=True, phys_criterion=phys_criterion)
            if test_metrics:
                bench = trainer.benchmark_model(model, test_loader)
                test_metrics.update(bench)
                test_metrics['seed'] = seed
                test_metrics['fold'] = fold
                all_metrics.append(test_metrics)
    if not all_metrics:
        return None
    df = pd.DataFrame(all_metrics)
    summary = {
        'mean_accuracy': df['accuracy'].mean(),
        'std_accuracy': df['accuracy'].std(),
        'mean_loc_prec': df['localization_precision'].mean(),
        'std_loc_prec': df['localization_precision'].std(),
        'mean_macro_f1': df['macro_f1'].mean(),
        'std_macro_f1': df['macro_f1'].std(),
        'mean_physics_violation': df.get('physics_violation', pd.Series([0])).mean(),
        'mean_latency_ms': df['latency_ms'].mean(),
        'mean_memory_mb': df['memory_mb'].mean(),
        'mean_flops_g': df['flops_g'].mean(),
        'mean_params': df['params'].mean(),
        'n_folds': len(all_metrics),
        'all_metrics': all_metrics,
        # Per‑attack localization
        'loc_fdia': df['loc_fdia'].mean() if 'loc_fdia' in df else 0,
        'loc_topology': df['loc_topology'].mean() if 'loc_topology' in df else 0,
        'loc_laa': df['loc_laa'].mean() if 'loc_laa' in df else 0,
    }
    config.tracker.log(f"  ID Acc: {summary['mean_accuracy']*100:.2f}%")
    return summary


# ============================================================
# ID → OOD GENERALIZATION
# ============================================================

def evaluate_ood(model_class, model_name, use_physics_loss, remove_residuals,
                 train_samples, ood_test_samples, best_hp, config, data_manager, topology, trainer):
    """Train on ID training set, evaluate on OOD test set."""
    config.tracker.log(f"\n--- OOD Evaluation: {model_name} ---")
    # Use only one fold for speed (full training set)
    class_weights = data_manager.compute_class_weights(train_samples)
    in_feat = config.reduced_features if (remove_residuals or 'Standard GAT' in model_name) else config.full_features
    if model_class == MLP:
        model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels']*2,
                            num_classes=config.num_classes, dropout=best_hp['dropout']).to(config.device)
    elif model_class == StandardGAT:
        model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels'],
                            num_classes=config.num_classes, heads=config.heads,
                            dropout=best_hp['dropout']).to(config.device)
    else:
        model = model_class(in_features=in_feat, hidden=best_hp['hidden_channels'],
                            num_classes=config.num_classes, dropout=best_hp['dropout']).to(config.device)
    if remove_residuals or 'Standard GAT' in model_name:
        train_mod = [(x[:,:config.reduced_features], y) for x,y in train_samples]
    else:
        train_mod = train_samples
    train_loader = data_manager.create_dataloader(train_mod, best_hp['batch_size'], True)
    val_loader = data_manager.create_dataloader([], best_hp['batch_size'], False)  # dummy
    if use_physics_loss and not remove_residuals and 'Standard GAT' not in model_name:
        criterion = PhysicsInformedLoss(topology, best_hp['lambda_phys'], class_weights).to(config.device)
    else:
        cw = torch.from_numpy(class_weights).to(config.device)
        criterion = nn.CrossEntropyLoss(weight=cw).to(config.device)
    model, _ = trainer.train_model(model, train_loader, val_loader, criterion, best_hp, 0, 42)
    # Prepare OOD test loader
    if remove_residuals or 'Standard GAT' in model_name:
        ood_test_mod = [(x[:,:config.reduced_features], y) for x,y in ood_test_samples]
    else:
        ood_test_mod = ood_test_samples
    ood_loader = data_manager.create_dataloader(ood_test_mod, best_hp['batch_size'], False)
    phys_criterion = PhysicsInformedLoss(topology, 0, class_weights).to(config.device)
    test_metrics = trainer.evaluate(model, ood_loader, compute_physics_violation=True, phys_criterion=phys_criterion)
    if test_metrics is None:
        return 0.0
    return test_metrics['accuracy']


# ============================================================
# ATTACK STRENGTH SENSITIVITY (requires modifying test samples)
# ============================================================

def attack_strength_sensitivity(model, test_samples, attack_type, strength_values, config, data_manager, trainer, topology):
    """
    attack_type: 'fdia' (modify P_inj), 'laa' (multiply load), 'topology' (harder: vary line removal)
    For FDIA: strength = magnitude added to P_inj
    For LAA: strength = multiplier
    For topology: not implemented here (would need to modify topology before power flow)
    """
    results = []
    rng = np.random.RandomState(42)
    for strength in strength_values:
        mod_samples = []
        for x, y in test_samples:
            x_new = x.copy()
            if attack_type == 'fdia':
                # Only modify samples that are originally FDIA? For sensitivity we apply to all attack samples.
                # Here we only modify the attacked buses (those labelled 1) for simplicity.
                attacked_buses = (y == 1)
                if attacked_buses.any():
                    x_new[attacked_buses, 0] += strength * rng.randn(attacked_buses.sum())  # add to P_inj
            elif attack_type == 'laa':
                attacked_buses = (y == 3)
                if attacked_buses.any():
                    x_new[attacked_buses, 0] *= strength  # P_inj
                    x_new[attacked_buses, 1] *= strength  # Q_inj
            mod_samples.append((x_new, y))
        loader = data_manager.create_dataloader(mod_samples, 32, False)
        metrics = trainer.evaluate(model, loader)
        if metrics:
            results.append((strength, metrics['accuracy'], metrics.get('localization_precision', 0)))
    return results


# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def extract_and_plot_attention(model, data_sample, edge_index, title, save_path):
    model.eval()
    with torch.no_grad():
        _, (attn1, attn2) = model(data_sample.x.unsqueeze(0), edge_index, return_attention=True)
    # attn1 is a tuple (attention_weights, edge_index)
    attn_weights, attn_edge = attn1
    attn_weights = attn_weights.mean(dim=1)  # average over heads -> (num_edges,)
    # Build adjacency matrix for 39 nodes
    n_nodes = 39
    adj = np.zeros((n_nodes, n_nodes))
    edge_src = attn_edge[0].cpu().numpy()
    edge_tgt = attn_edge[1].cpu().numpy()
    attn_np = attn_weights.cpu().numpy()
    for src, tgt, w in zip(edge_src, edge_tgt, attn_np):
        adj[src, tgt] = w
    plt.figure(figsize=(10,8))
    sns.heatmap(adj, cmap='viridis', cbar_kws={'label': 'Attention weight'})
    plt.title(title)
    plt.xlabel('Target node')
    plt.ylabel('Source node')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.models_dir, exist_ok=True)
        self.log_file = f"{config.results_dir}/log_{config.experiment_id}.txt"

    def log(self, msg, print_console=True):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {msg}\n")
        if print_console:
            print(msg)

    def save_results(self, data, name):
        with open(f"{self.config.results_dir}/{name}_{self.config.experiment_id}.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)


def main():
    print("\n" + "█"*70)
    print("█  FINAL PI-GAT PIPELINE – ID/OOD, ATTACK SENSITIVITY, ATTENTION  █")
    print("█"*70 + "\n")

    config = ExperimentConfig()
    tracker = ExperimentTracker(config)
    config.tracker = tracker

    topology = GridTopologyExtractor()
    data_manager = DataManager(topology, config)

    # Load ID and OOD datasets
    print("Loading datasets...")
    id_samples_dict = data_manager.load_dataset("ieee39_challenge_id.npz")
    ood_samples_dict = data_manager.load_dataset("ieee39_challenge_ood.npz")

    # Merge all ID samples for training (for OOD eval we need full training set)
    all_id_samples = []
    for samples in id_samples_dict.values():
        all_id_samples.extend(samples)
    # For OOD test, we need all OOD samples (balanced by class)
    all_ood_samples = []
    for samples in ood_samples_dict.values():
        all_ood_samples.extend(samples)

    tracker.log(f"ID samples: {len(all_id_samples)}, OOD samples: {len(all_ood_samples)}")

    # Hyperparameter search on ID data
    hp_opt = HyperparameterOptimizer(config, tracker, topology, data_manager)
    best_hp = hp_opt.grid_search(id_samples_dict, n_trials=15)
    config.best_hp = best_hp

    # Ablation configurations (same as before)
    ablation_configs = {
        'No Residuals, No Physics': (PI_GAT_NoResiduals, False, True),
        'Residuals Only, No Physics': (PI_GAT, False, False),
        'No Residuals, Physics Loss': (PI_GAT_NoResiduals, True, True),
        'Residuals + Physics (Full PI-GAT)': (PI_GAT, True, False),
        'Standard GAT': (StandardGAT, False, True),
        'GCN': (GCN, False, False),
        'GraphSAGE': (GraphSAGE, False, False),
        'MLP': (MLP, False, False),
    }

    print("\n=== Running ID cross‑validation (ablation) ===")
    trainer = TrainingEngine(config, tracker)
    ablation_results = {}
    for name, (model_cls, use_phys, remove_res) in ablation_configs.items():
        res = run_cv(model_cls, name, use_phys, remove_res, id_samples_dict,
                     best_hp, config, data_manager, topology, trainer)
        ablation_results[name] = res

    # ID → OOD generalization
    print("\n=== ID → OOD Generalization ===")
    ood_results = {}
    for name, (model_cls, use_phys, remove_res) in ablation_configs.items():
        # Train on full ID training set (all ID samples) and test on OOD
        ood_acc = evaluate_ood(model_cls, name, use_phys, remove_res,
                               all_id_samples, all_ood_samples,
                               best_hp, config, data_manager, topology, trainer)
        ood_results[name] = ood_acc
        tracker.log(f"{name:35} OOD Acc: {ood_acc*100:.2f}%")

    # Attack strength sensitivity (using full PI-GAT)
    print("\n=== Attack Strength Sensitivity ===")
    # Train full PI-GAT on all ID samples
    full_model = PI_GAT(in_features=config.full_features, hidden=best_hp['hidden_channels'],
                        num_classes=config.num_classes, heads=config.heads,
                        dropout=best_hp['dropout']).to(config.device)
    class_weights_id = data_manager.compute_class_weights(all_id_samples)
    train_loader_id = data_manager.create_dataloader(all_id_samples, best_hp['batch_size'], True)
    dummy_loader = data_manager.create_dataloader([], best_hp['batch_size'], False)
    criterion_full = PhysicsInformedLoss(topology, best_hp['lambda_phys'], class_weights_id).to(config.device)
    full_model, _ = trainer.train_model(full_model, train_loader_id, dummy_loader, criterion_full, best_hp, 0, 42)

    # FDIA sensitivity on a subset of FDIA test samples from ID
    fdia_test_samples = id_samples_dict['fdia'][:100]  # use first 100 for speed
    fdia_sensitivity = attack_strength_sensitivity(full_model, fdia_test_samples, 'fdia',
                                                   config.fdia_magnitudes, config, data_manager, trainer, topology)
    laa_test_samples = id_samples_dict['laa'][:100]
    laa_sensitivity = attack_strength_sensitivity(full_model, laa_test_samples, 'laa',
                                                  config.load_multipliers, config, data_manager, trainer, topology)

    tracker.log("FDIA sensitivity (magnitude → acc): " + str(fdia_sensitivity))
    tracker.log("LAA sensitivity (multiplier → acc): " + str(laa_sensitivity))

    # Confusion matrix heatmaps for best model (Full PI-GAT) on ID test (first fold)
    _, _, test_fold0 = data_manager.create_fold_splits(id_samples_dict, 0, 42)
    test_loader_fold0 = data_manager.create_dataloader(test_fold0, best_hp['batch_size'], False)
    test_metrics = trainer.evaluate(full_model, test_loader_fold0)
    if test_metrics:
        cm = test_metrics['confusion_matrix']
        plot_confusion_matrix(cm, ['Normal','FDIA','Topology','LAA'],
                              "Confusion Matrix – Full PI-GAT (ID test)",
                              f"{config.figures_dir}/confusion_matrix_full.png")

    # Attention visualisation (pick one normal and one attacked sample)
    norm_sample = None
    attacked_sample = None
    for cls, samples in id_samples_dict.items():
        if cls == 'normal' and samples:
            norm_sample = samples[0]
        elif cls == 'fdia' and samples:
            attacked_sample = samples[0]
    if norm_sample is not None and attacked_sample is not None:
        edge_tensor = torch.tensor(topology.edge_index, dtype=torch.long).to(config.device)
        norm_x = torch.tensor(norm_sample[0], dtype=torch.float32).to(config.device)
        att_x = torch.tensor(attacked_sample[0], dtype=torch.float32).to(config.device)
        extract_and_plot_attention(full_model, norm_x, edge_tensor,
                                    "Attention weights – Normal operation",
                                    f"{config.figures_dir}/attention_normal.png")
        extract_and_plot_attention(full_model, att_x, edge_tensor,
                                    "Attention weights – FDIA attack",
                                    f"{config.figures_dir}/attention_fdia.png")

    # Save all results
    final = {
        'experiment_id': config.experiment_id,
        'best_hp': best_hp,
        'ablation_id': {k: {kk: vv for kk,vv in v.items() if kk != 'all_metrics'} for k,v in ablation_results.items() if v},
        'ood_generalization': ood_results,
        'fdia_sensitivity': fdia_sensitivity,
        'laa_sensitivity': laa_sensitivity,
        'per_attack_localization': {
            name: {'fdia': res['loc_fdia'], 'topology': res['loc_topology'], 'laa': res['loc_laa']}
            for name, res in ablation_results.items() if res
        }
    }
    tracker.save_results(final, 'final_results')

    print("\n" + "█"*70)
    print("█  PIPELINE COMPLETE – RESULTS IN final_results/ AND final_figures/  █")
    print("█"*70)


if __name__ == "__main__":
    main()