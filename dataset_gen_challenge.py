#!/usr/bin/env python3
"""
Challenging Power Grid Dataset Generator
----------------------------------------
Generates a non-trivial dataset for GNN-based attack detection.
Features (6 per node): [p_inj, q_inj, v, theta, r_p, r_q]
No bus type encoding – models must learn from physics and residuals.

Output: (39, 6) features per sample, node labels 0-3.
"""

import math
import warnings
from typing import List, Tuple
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as pt
import networkx as nx
from scipy.linalg import cholesky, eigh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ChallengingDataEngine:
    """
    Generates a challenging dataset with:
    - No trivial features (bus types removed)
    - Higher measurement noise
    - Subtler attacks
    - Realistic load variations
    """

    def __init__(self, timesteps_per_class: int = 2000, seed: int = 2026):
        self.timesteps = timesteps_per_class
        self.seed = seed

        self.net = pn.case39()
        self.s_base = self.net.sn_mva
        self.n_buses = len(self.net.bus)
        self.slack_idx = self.net.ext_grid.bus.iloc[0]

        # Topology and admittance
        self.edge_index, self.G_line, self.B_line = self._build_admittance_and_topology()

        # DC Jacobian for FDIA
        self.H_dc = self._build_dc_jacobian()

        # Spatial correlation matrix for load noise
        self.L_spatial = self._build_spatial_covariance()

    # ------------------------------------------------------------------
    # Internal builders (same as before, keep for compatibility)
    # ------------------------------------------------------------------

    def _build_admittance_and_topology(self):
        edges_fb, edges_tb = [], []
        g_vals, b_vals = [], []

        for _, line in self.net.line.iterrows():
            fb, tb = int(line.from_bus), int(line.to_bus)
            zb = (self.net.bus.vn_kv.at[fb] ** 2) / self.s_base
            r = (line.r_ohm_per_km * line.length_km) / zb
            x = (line.x_ohm_per_km * line.length_km) / zb
            y = 1.0 / (r + 1j * x)
            edges_fb.extend([fb, tb])
            edges_tb.extend([tb, fb])
            g_vals.extend([y.real, y.real])
            b_vals.extend([y.imag, y.imag])

        for _, trafo in self.net.trafo.iterrows():
            fb, tb = int(trafo.hv_bus), int(trafo.lv_bus)
            r = (trafo.vkr_percent / 100.0) * (self.s_base / trafo.sn_mva)
            x = (np.sqrt(trafo.vk_percent**2 - trafo.vkr_percent**2) / 100.0) * (self.s_base / trafo.sn_mva)
            y = 1.0 / (r + 1j * x)
            edges_fb.extend([fb, tb])
            edges_tb.extend([tb, fb])
            g_vals.extend([y.real, y.real])
            b_vals.extend([y.imag, y.imag])

        edge_index = np.vstack([edges_fb, edges_tb])
        return edge_index, np.array(g_vals, dtype=np.float32), np.array(b_vals, dtype=np.float32)

    def _build_dc_jacobian(self) -> np.ndarray:
        n_lines = len(self.net.line)
        Bbus = np.zeros((self.n_buses, self.n_buses))
        Bline = np.zeros((n_lines, self.n_buses))

        for idx, line in self.net.line.iterrows():
            zb = (self.net.bus.vn_kv.at[line.from_bus] ** 2) / self.s_base
            x = (line.x_ohm_per_km * line.length_km) / zb
            b = 1.0 / (x + 1e-9)
            fb, tb = int(line.from_bus), int(line.to_bus)
            Bline[idx, fb] = b
            Bline[idx, tb] = -b
            Bbus[fb, tb] -= b
            Bbus[tb, fb] -= b
            Bbus[fb, fb] += b
            Bbus[tb, tb] += b

        H = np.vstack([Bbus, Bline])
        return np.delete(H, self.slack_idx, axis=1)

    def _build_spatial_covariance(self, length_scale: float = 2.0) -> np.ndarray:
        G = pt.create_nxgraph(self.net)
        try:
            dist = dict(nx.all_pairs_shortest_path_length(G))
        except nx.NetworkXError:
            dist = {}
            for i in range(self.n_buses):
                dist[i] = {}
                for j in range(self.n_buses):
                    if i == j:
                        dist[i][j] = 0
                    elif nx.has_path(G, i, j):
                        dist[i][j] = nx.shortest_path_length(G, i, j)
                    else:
                        dist[i][j] = 10

        Sigma = np.zeros((self.n_buses, self.n_buses))
        for i in range(self.n_buses):
            for j in range(self.n_buses):
                d = dist.get(i, {}).get(j, 10)
                Sigma[i, j] = np.exp(-(d**2) / (2 * length_scale**2))

        # Regularization
        reg = 1e-4
        Sigma_reg = Sigma + reg * np.eye(self.n_buses)
        try:
            L = cholesky(Sigma_reg, lower=True)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = eigh(Sigma_reg)
            eigvals = np.maximum(eigvals, 1e-6)
            Sigma_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = cholesky(Sigma_fixed, lower=True)
        return L

    def _ac_power_injections(self, v: np.ndarray, theta: np.ndarray):
        p_calc = np.zeros(self.n_buses, dtype=np.float64)
        q_calc = np.zeros(self.n_buses, dtype=np.float64)
        for e in range(self.edge_index.shape[1]):
            i = self.edge_index[0, e]
            j = self.edge_index[1, e]
            g = self.G_line[e]
            b = self.B_line[e]
            dtheta = theta[i] - theta[j]
            vi, vj = v[i], v[j]
            cos_d = math.cos(dtheta)
            sin_d = math.sin(dtheta)
            p_flow = g * vi**2 - vi * vj * (g * cos_d + b * sin_d)
            q_flow = -b * vi**2 - vi * vj * (g * sin_d - b * cos_d)
            p_calc[i] += p_flow
            q_calc[i] += q_flow
        return p_calc.astype(np.float32), q_calc.astype(np.float32)

    # ------------------------------------------------------------------
    # Main generation method (HARDER SETTINGS)
    # ------------------------------------------------------------------

    def generate_time_series(self, attack_class: int, ood: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
        rng = np.random.RandomState(self.seed + attack_class + (100 if ood else 0))

        base_p = self.net.load.p_mw.values.copy()
        base_q = self.net.load.q_mvar.values.copy()
        load_buses = self.net.load.bus.values
        n_loads = len(self.net.load)

        eta = np.zeros(self.n_buses, dtype=np.float64)
        samples = []

        # CHALLENGING PARAMETERS
        if ood:
            load_range = (0.5, 1.5)          # wider but still feasible
            noise_std = 0.10                 # higher noise
            attack_multiplier = (1.5, 2.2)   # subtler load attack
            fdia_magnitude = (-0.05, 0.05)   # smaller FDIA
            diurnal_factor = 0.15
        else:
            load_range = (0.7, 1.3)
            noise_std = 0.08                 # increased from 0.04
            attack_multiplier = (1.8, 2.5)   # reduced from (2.0,3.5)
            fdia_magnitude = (-0.04, 0.04)   # reduced from (-0.06,0.06)
            diurnal_factor = 0.12

        max_attempts = 40
        failed = 0
        skipped = 0

        for t in range(self.timesteps):
            for attempt in range(max_attempts):
                try:
                    # Reset network
                    self.net.load.p_mw = base_p.copy()
                    self.net.load.q_mvar = base_q.copy()
                    self.net.line.in_service = True

                    # Temporal variation
                    diurnal = rng.uniform(load_range[0], load_range[1]) + diurnal_factor * math.sin(2 * math.pi * t / 96)

                    # Spatially correlated noise (higher std)
                    z = rng.normal(0, noise_std, self.n_buses)
                    eta = 0.9 * eta + self.L_spatial @ z

                    new_p = base_p * (diurnal + eta[load_buses])
                    new_q = base_q * (diurnal + eta[load_buses])
                    new_p = np.clip(new_p, base_p * 0.5, base_p * 1.5)
                    new_q = np.clip(new_q, base_q * 0.5, base_q * 1.5)
                    self.net.load.p_mw = new_p
                    self.net.load.q_mvar = new_q

                    y = np.zeros(self.n_buses, dtype=np.int64)
                    attack_active = t > self.timesteps // 3
                    target_bus = None

                    if attack_class == 1 and attack_active:
                        candidates = [b for b in range(self.n_buses) if b != self.slack_idx]
                        if len(candidates) > 0:
                            target_bus = rng.choice(candidates)
                            y[target_bus] = 1

                    elif attack_class == 2 and attack_active:
                        in_service = np.where(self.net.line.in_service)[0]
                        if len(in_service) > 0:
                            line_idx = rng.choice(in_service)
                            self.net.line.in_service.at[line_idx] = False
                            fb = int(self.net.line.from_bus.at[line_idx])
                            tb = int(self.net.line.to_bus.at[line_idx])
                            y[fb] = 2
                            y[tb] = 2

                    elif attack_class == 3 and attack_active:
                        load_idx = rng.choice(n_loads)
                        multiplier = rng.uniform(attack_multiplier[0], attack_multiplier[1])
                        self.net.load.p_mw.at[load_idx] *= multiplier
                        self.net.load.q_mvar.at[load_idx] *= multiplier
                        target_bus = int(self.net.load.bus.at[load_idx])
                        y[target_bus] = 3

                    pp.runpp(self.net, solver='nr', init='dc', max_iteration=50, tolerance_mva=1e-4)
                    if not self.net.converged:
                        if attempt == max_attempts - 1:
                            skipped += 1
                        continue

                    v = self.net.res_bus.vm_pu.values.astype(np.float32)
                    theta = np.radians(self.net.res_bus.va_degree.values).astype(np.float32)
                    p_inj = (self.net.res_bus.p_mw.values / self.s_base).astype(np.float32)
                    q_inj = (self.net.res_bus.q_mvar.values / self.s_base).astype(np.float32)

                    # Stealth FDIA injection (small magnitude)
                    if attack_class == 1 and attack_active and target_bus is not None:
                        if target_bus < self.slack_idx:
                            reduced_idx = target_bus
                        elif target_bus > self.slack_idx:
                            reduced_idx = target_bus - 1
                        else:
                            reduced_idx = None
                        if reduced_idx is not None:
                            attack_magnitude = rng.uniform(fdia_magnitude[0], fdia_magnitude[1])
                            c_vec = np.zeros(self.n_buses - 1)
                            c_vec[reduced_idx] = attack_magnitude
                            a_vec = self.H_dc @ c_vec
                            p_inj += a_vec[:self.n_buses].astype(np.float32)

                    # Residuals (physics inconsistency)
                    p_calc, q_calc = self._ac_power_injections(v, theta)
                    r_p = np.clip(p_inj - p_calc, -0.5, 0.5)
                    r_q = np.clip(q_inj - q_calc, -0.5, 0.5)

                    # FEATURES: Only 6 columns – NO bus type encoding!
                    x = np.column_stack([
                        p_inj, q_inj, v, theta,
                        r_p, r_q
                    ]).astype(np.float32)

                    samples.append((x, y))
                    break  # success

                except Exception:
                    if attempt == max_attempts - 1:
                        failed += 1
                    continue

            if (t + 1) % 200 == 0 or t == self.timesteps - 1:
                print(f"  Class {attack_class}: {len(samples)}/{self.timesteps} samples (skip={skipped}, fail={failed})", end='\r')

        print(f"  Class {attack_class} completed: {len(samples)}/{self.timesteps} samples")
        return samples

    def generate_full_dataset(self, ood: bool = False) -> dict:
        print(f"\nGenerating {'OOD' if ood else 'in-distribution'} dataset (challenging)...")
        print(f"  Target samples per class: {self.timesteps}")
        print(f"  Features per node: 6 (P, Q, V, theta, r_p, r_q) – no bus types")
        dataset = {
            'normal': self.generate_time_series(0, ood),
            'fdia': self.generate_time_series(1, ood),
            'topology': self.generate_time_series(2, ood),
            'laa': self.generate_time_series(3, ood),
        }
        for name, data in dataset.items():
            print(f"    {name:10s}: {len(data)} valid samples")
        return dataset

    def save_dataset(self, dataset: dict, filepath: str):
        save_dict = {}
        for cls, samples in dataset.items():
            if not samples:
                continue
            x_list = [s[0] for s in samples]
            y_list = [s[1] for s in samples]
            save_dict[f'{cls}_x'] = np.stack(x_list, axis=0)
            save_dict[f'{cls}_y'] = np.stack(y_list, axis=0)
        if save_dict:
            np.savez_compressed(filepath, **save_dict)
            print(f"\nDataset saved to {filepath}")
            for k, v in save_dict.items():
                print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Challenging Power Grid Dataset Generator (6 features, no bus types)")
    print("=" * 60)

    engine = ChallengingDataEngine(timesteps_per_class=2000, seed=2026)

    print("\nGenerating in-distribution dataset...")
    id_dataset = engine.generate_full_dataset(ood=False)
    engine.save_dataset(id_dataset, "ieee39_challenge_id.npz")

    print("\nGenerating OOD dataset (more extreme) ...")
    ood_dataset = engine.generate_full_dataset(ood=True)
    engine.save_dataset(ood_dataset, "ieee39_challenge_ood.npz")

    print("\n" + "=" * 60)
    print("Generation complete. Use ieee39_challenge_id.npz for main experiments.")
    print("=" * 60)