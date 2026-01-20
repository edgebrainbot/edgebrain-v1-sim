# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import csv
import torch
import pickle
from pathlib import Path
from edge_drl_student.utils.utils import ActionMode
from typing import Dict, Generator, Tuple, Optional, Union
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import Transition, TransitionBatch


class SoloReplayBuffer(BaseBuffer):
    """
    SoloReplayBuffer
    ----------------
    A united replay buffer implementation built on top of the base class.
    """

    def __init__(self, 
                 buffer_size: int, 
                 prefill_size: int,
                 sampler: Optional[UniformSampler] = None, 
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 name: str = "SoloReplayBuffer",
                 device: Optional[str] = 'cpu'):
        if sampler is None:
            sampler = UniformSampler(buffer_size, random_seed=42, device=device)
            
        super().__init__(buffer_size, prefill_size, sampler, name, device=device)

        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add_transition(self, t: Transition, action_mode: ActionMode = ActionMode.UNCERTAIN):
        """
        Add one transition into the buffer.
        Typically called at every environment step.
        """
        idx = self._idx

        self.obs[idx] = t.obs
        self.actions[idx] = t.action
        self.rewards[idx] = t.reward
        self.next_obs[idx] = t.next_obs
        self.dones[idx] = t.done

        self._idx = (self._idx + 1) % self.buffer_size  # Circular buffer
        self._history_count += 1                        # Total transitions added
        
        if self._idx == 0 and not self._full:
            self._full = True

    def _get_samples(self, indices: torch.Tensor) -> TransitionBatch:
        """
        Given sampled indices, return the corresponding transitions.
        """
        return TransitionBatch(
            obs=self.obs[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_obs[indices],
            done=self.dones[indices]
        )
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A batch of transitions obtained via `_get_samples()`.
        """
        valid_size = self.size
        if valid_size <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) exceeds buffer_size ({self.buffer_size})."
            )
        if not self.is_ready:
            raise RuntimeError(
                f"Buffer not prefilled: current size is {self.size}, "
                f"required prefill size is {self.prefill_size}."
            )

        # Delegate sampling to sampler
        sample_indices = self.sampler.fetch(batch_size, valid_size)
        return self._get_samples(sample_indices)
    
    
    def get_samples(self, indices: torch.Tensor) -> TransitionBatch:
        """Public API to get samples by indices (reserved for external use)."""
        return self._get_samples(indices)

    
    # ======================================================
    # Dump / Load Public API
    # ======================================================
    def dump(self, path: Union[str, Path], format: str = 'torch', compress: bool = False):
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'torch':
            self._dump_torch(path, compress)
        elif format == 'pickle':
            self._dump_pickle(path)
        elif format == 'csv':
            self._dump_csv(path)
        else:
            raise ValueError(f"Unknown dump format {format}")

        print(f"Saved buffer data to {path}")

    def load(self, 
             path: Union[str, Path],
             format: str = 'auto',
             device: Optional[str] = None):
        """Load buffer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        # auto detect
        if format == 'auto':
            ext = path.suffix.lower()
            if ext in ['.pt', '.pth']:
                format = 'torch'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            elif ext == '.csv':
                format = 'csv'
            else:
                raise ValueError(f"Could not auto-detect format from {ext}")

        # dispatch
        if format == 'torch':
            self._load_torch(path, device)
        elif format == 'pickle':
            self._load_pickle(path)
        elif format == 'csv':
            self._load_csv(path)
        else:
            raise ValueError(f"Unknown load format {format}")

        print(f"Loaded buffer ← {path}")

    # ======================================================
    # TORCH FORMAT
    # ======================================================
    def _dump_torch(self, path: Path, compress: bool):
        state = {
            # data
            'obs': self.obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_obs': self.next_obs,
            'dones': self.dones,

            # metadata
            'idx': self._idx,
            'full': self._full,
            'history_count': self._history_count,

            'buffer_size': self.buffer_size,
            'prefill_size': self.prefill_size,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,

            'sampler_state': getattr(self.sampler, "state_dict", lambda: {})()
        }

        torch.save(
            state,
            path,
            _use_new_zipfile_serialization=compress
        )

    def _load_torch(self, path: Path, device):
        data = torch.load(path, map_location=device or self.device)

        # restore tensors
        self.obs = data['obs'].to(self.device)
        self.actions = data['actions'].to(self.device)
        self.rewards = data['rewards'].to(self.device)
        self.next_obs = data['next_obs'].to(self.device)
        self.dones = data['dones'].to(self.device)

        # metadata
        self._idx = data['idx']
        self._full = data['full']
        self._history_count = data.get("history_count", self._history_count)

        # sanity restore dimensions
        self.obs_dim = data['obs_dim']
        self.action_dim = data['action_dim']

        # sampler state restore
        if hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(data["sampler_state"])

    # ======================================================
    # PICKLE FORMAT
    # ======================================================
    def _dump_pickle(self, path: Path):
        arr = lambda x: x.cpu().numpy()

        state = {
            'obs': arr(self.obs),
            'actions': arr(self.actions),
            'rewards': arr(self.rewards),
            'next_obs': arr(self.next_obs),
            'dones': arr(self.dones),

            'idx': self._idx,
            'full': self._full,
            'history_count': self._history_count,

            'buffer_size': self.buffer_size,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.obs = torch.tensor(data['obs'], device=self.device)
        self.actions = torch.tensor(data['actions'], device=self.device)
        self.rewards = torch.tensor(data['rewards'], device=self.device)
        self.next_obs = torch.tensor(data['next_obs'], device=self.device)
        self.dones = torch.tensor(data['dones'], device=self.device)

        self._idx = data['idx']
        self._full = data['full']
        self._history_count = data.get("history_count", 0)

        self.obs_dim = data['obs_dim']
        self.action_dim = data['action_dim']

    # ==========================================================
    # CSV Dump / Load
    # ==========================================================

    def _dump_csv(self, path: Path):
        """
        Save buffer as CSV.
        Each row = one transition.
        Format:
            obs..., action..., reward, next_obs..., done
        """
        valid = self.size

        with open(path, "w", newline='') as f:
            writer = csv.writer(f)

            # ---- Header ----
            header = []
            header += [f"obs_{i}" for i in range(self.obs.shape[1])]
            header += [f"action_{i}" for i in range(self.actions.shape[1])]
            header += ["reward"]
            header += [f"next_obs_{i}" for i in range(self.next_obs.shape[1])]
            header += ["done"]
            writer.writerow(header)

            # ---- Rows ----
            for i in range(valid):
                row = []
                row += self.obs[i].cpu().tolist()
                row += self.actions[i].cpu().tolist()
                row += [float(self.rewards[i].item())]
                row += self.next_obs[i].cpu().tolist()
                row += [bool(self.dones[i].item())]
                writer.writerow(row)

    def _load_csv(self, path: Path):
        """
        Load transitions from CSV and rebuild buffer.
        Expect CSV columns in same order as dump.
        """
        self.reset()

        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            obs_dim = self.obs.shape[1]
            action_dim = self.actions.shape[1]
            next_obs_dim = self.next_obs.shape[1]

            for row in reader:
                # CSV row is a list of strings
                row = [float(x) if x not in ["True", "False"] else (x == "True") for x in row]

                # Slicing row into components
                o = torch.tensor(row[0:obs_dim], dtype=torch.float32, device=self.device)
                a = torch.tensor(row[obs_dim:obs_dim + action_dim], dtype=torch.float32,
                                 device=self.device)
                r = torch.tensor([row[obs_dim + action_dim]], dtype=torch.float32,
                                 device=self.device)

                no_start = obs_dim + action_dim + 1
                no_end = no_start + next_obs_dim

                no = torch.tensor(row[no_start:no_end], dtype=torch.float32, device=self.device)
                d = torch.tensor([row[-1]], dtype=torch.float32, device=self.device)

                t = Transition(o, a, r, no, d)
                self.add_transition(t)

    # ======================================================
    # Utilities
    # ======================================================
    def get_statistics(self) -> Dict[str, float]:
        valid = self.size

        if valid == 0:
            return {
                "size": 0,
                "capacity": self.buffer_size,
                "utilization": 0.0,
            }

        return {
            "size": valid,
            "capacity": self.buffer_size,
            "history_count": self._history_count,
            "utilization": valid / self.buffer_size,
            "mean_reward": float(self.rewards[:valid].mean()),
            "std_reward": float(self.rewards[:valid].std()),
            "done_ratio": float(self.dones[:valid].mean())
        }

    def __repr__(self):
        return (
            f"SoloReplayBuffer(size={self.size}/{self.buffer_size}, "
            f"ready={self.is_ready}, "
            f"history={self._history_count}, "
            f"device={self.device})"
        )
        
        
def test_dump_and_load():

    # ======================================================
    # 1. create buffer
    # ======================================================
    buf = SoloReplayBuffer(
        buffer_size=10,
        prefill_size=3,
        obs_dim=4,
        action_dim=2,
        device="cpu"
    )

    print("Initial buffer:", buf)

    # ======================================================
    # 2. write in transition
    # ======================================================
    for i in range(12):
        t = Transition(
            obs=torch.tensor([i, i+1, i+2, i+3], dtype=torch.float32),
            action=torch.tensor([i * 0.1, i * 0.2]),
            reward=torch.tensor([i * 1.0]),
            next_obs=torch.tensor([i+0.1, i+1.1, i+2.1, i+3.1]),
            done=torch.tensor([i % 2 == 0], dtype=torch.float32)
        )
        buf.add_transition(t)

    print("\nAfter adding 12 transitions:")
    print(buf.get_statistics())

    # ======================================================
    # 3. Dump to 3 formats
    # ======================================================
    out_dir = Path("./buffer_test_output")
    out_dir.mkdir(exist_ok=True)

    path_pt = out_dir / "buffer.pt"
    path_pkl = out_dir / "buffer.pkl"
    path_csv = out_dir / "buffer.csv"

    buf.dump(path_pt, format="torch")
    buf.dump(path_pkl, format="pickle")
    buf.dump(path_csv, format="csv")
    
    # ======================================================
    # 4. Load back into new buffers
    # ======================================================
    buf_pt = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_pt.load(path_pt)

    buf_pkl = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_pkl.load(path_pkl)

    buf_csv = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_csv.load(path_csv)

    # ======================================================
    # 5. Compare data to verify correctness
    # ======================================================
    assert torch.allclose(buf.obs, buf_pt.obs)
    assert torch.allclose(buf.actions, buf_pt.actions)
    assert torch.allclose(buf.rewards, buf_pt.rewards)
    assert torch.allclose(buf.next_obs, buf_pt.next_obs)
    assert torch.allclose(buf.dones, buf_pt.dones)

    assert torch.allclose(buf.obs, buf_pkl.obs)
    assert torch.allclose(buf.obs, buf_csv.obs)

    print("\nAll formats loaded correctly!")
    print("✓ torch OK")
    print("✓ pickle OK")
    print("✓ csv OK")

    print("\nPT statistics:", buf_pt.get_statistics())
    print("PKL statistics:", buf_pkl.get_statistics())
    print("CSV statistics:", buf_csv.get_statistics())


if __name__ == "__main__":
    test_dump_and_load()
    print("\n==== TEST PASSED ====")