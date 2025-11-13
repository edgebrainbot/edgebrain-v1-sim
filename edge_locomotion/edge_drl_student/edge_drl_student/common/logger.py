import os
import time
import statistics
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Lightweight RL logger inspired by rsl_rl OnPolicyRunner.log(),
    but simplified and adapted for off-policy training loops.

    Features:
    - TensorBoard logging
    - Pretty console printout
    - reward buffer, episode length buffer
    - loss metrics logging
    - performance statistics (FPS, iteration time)
    """

    def __init__(self, log_dir: str, flush_secs=10, device="cpu"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.device = device

        # counters
        self.total_time = 0.0
        self.total_env_steps = 0
        self.current_iteration = 0

        # episode statistics
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

        print(f"[EdgeDLLogger] Logging to: {self.log_dir}")

    # ------------------------------------------------------------------ #
    # Logging API
    # ------------------------------------------------------------------ #

    def add_episode_info(self, reward, length):
        """Add completed episode stats."""
        self.rewbuffer.append(float(reward))
        self.lenbuffer.append(float(length))

    def log_losses(self, loss_dict: dict, step: int):
        """Log loss scalars."""
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, step)

    def log_train_metrics(self, step: int):
        """Log reward & episode length."""
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), step)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), step)

    def log_policy_stats(self, noise_std: float, step: int):
        """Log policy noise/entropy/etc."""
        self.writer.add_scalar("Policy/noise_std", noise_std, step)

    def log_performance(self, fps: float, step: int, collection_time: float, learn_time: float):
        """Log speed and timing info."""
        self.writer.add_scalar("Perf/fps", fps, step)
        self.writer.add_scalar("Perf/collection_time", collection_time, step)
        self.writer.add_scalar("Perf/learn_time", learn_time, step)

    # ------------------------------------------------------------------ #
    # Pretty Print like rsl_rl
    # ------------------------------------------------------------------ #

    def print_summary(
        self,
        it: int,
        total_iterations: int,
        fps: float,
        loss_dict: dict,
        noise_std: float,
        collection_time: float,
        learn_time: float
    ):
        pad = 32
        width = 70
        s = f" Learning iteration {it}/{total_iterations} "

        msg = (
            f"\n{'#' * width}\n"
            f"{s.center(width)}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection {collection_time:.3f}s, learn {learn_time:.3f}s)\n"
            f"{'Policy noise std:':>{pad}} {noise_std:.3f}\n"
        )

        for key, val in loss_dict.items():
            msg += f"{('Loss ' + key + ':'):>{pad}} {val:.4f}\n"

        if len(self.rewbuffer) > 0:
            msg += f"{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"
            msg += f"{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"

        msg += (
            f"{'-' * width}\n"
            f"{'Total env steps:':>{pad}} {self.total_env_steps}\n"
            f"{'Total time:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.total_time))}\n"
        )

        print(msg)

    # ------------------------------------------------------------------ #
    # Main entry called by training loop
    # ------------------------------------------------------------------ #

    def log_iteration(
        self,
        *,
        it: int,
        total_iterations: int,
        loss_dict: dict,
        noise_std: float,
        env_steps_this_iter: int,
        collection_time: float,
        learn_time: float
    ):
        """Main logging entry point (similar to rsl_rl runner.log)."""

        # update counters
        self.current_iteration = it
        self.total_env_steps += env_steps_this_iter
        iter_time = collection_time + learn_time
        self.total_time += iter_time
        fps = env_steps_this_iter / iter_time

        # Write to TF summary
        self.log_losses(loss_dict, it)
        self.log_train_metrics(it)
        self.log_policy_stats(noise_std, it)
        self.log_performance(fps, it, collection_time, learn_time)

        # Print to console
        self.print_summary(
            it=it,
            total_iterations=total_iterations,
            fps=fps,
            loss_dict=loss_dict,
            noise_std=noise_std,
            collection_time=collection_time,
            learn_time=learn_time,
        )

    def close(self):
        self.writer.close()
