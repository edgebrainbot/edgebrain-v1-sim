import os
import torch
import warnings
import torch.nn as nn
from edge_drl_student.agents.off_policy_agent import OffPolicyAgent
from edge_drl_student.memory.buffers.base_buffer import TransitionBatch
from edge_drl_student.networks.actor_critic import ActorCritic, ActorNetwork, CriticNetwork
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.common.normalizer import BaseNormalizer, MinMaxNormalizer, EmpiricalNormalizer
from edge_drl_student.common.noise import Noise, OUNoise, GaussianNoise
from typing import Union
import numpy as np

class RunningMeanStd:
    """运行时均值和标准差计算"""
    def __init__(self, epsilon=1e-4, device="cpu"):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = epsilon
        self.device = device
        
    def update(self, x):
        """更新统计量"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if hasattr(x, '__len__') else 1
        
        delta = batch_mean - self.mean.item()
        tot_count = self.count + batch_count
        
        new_mean = self.mean.item() + delta * batch_count / tot_count
        m_a = self.var.item() * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = torch.tensor(new_mean, device=self.device)
        self.var = torch.tensor(new_var, device=self.device)
        self.count = tot_count
    
    def state_dict(self):
        return {
            'mean': self.mean.item(),
            'var': self.var.item(),
            'count': self.count
        }
    
    def load_state_dict(self, state_dict):
        self.mean = torch.tensor(state_dict['mean'], device=self.device)
        self.var = torch.tensor(state_dict['var'], device=self.device)
        self.count = state_dict['count']

class RewardNormalizer:
    """奖励归一化器"""
    def __init__(self, gamma=0.99, clip_range=10.0, device="cpu"):
        """
        Args:
            gamma: 折扣因子
            clip_range: 归一化后的裁剪范围
            device: 设备
        """
        self.rms = RunningMeanStd(device=device)
        self.gamma = gamma
        self.clip_range = clip_range
        self.return_ = 0.0
        self.episode_rewards = []
        self.device = device
        
    def update(self, reward):
        """更新统计量（收集阶段调用）"""
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self.episode_rewards.append(reward)
        self.return_ = self.return_ * self.gamma + reward
        
    def finish_episode(self):
        """Episode结束时更新RMS"""
        if len(self.episode_rewards) > 0:
            self.rms.update(self.episode_rewards)
            self.episode_rewards = []
        self.return_ = 0.0
    
    def normalize(self, rewards):
        """归一化奖励（训练阶段调用）"""
        if isinstance(rewards, torch.Tensor):
            mean = self.rms.mean
            std = torch.sqrt(self.rms.var + 1e-8)
            normalized = (rewards - mean) / std
            if self.clip_range is not None:
                normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
            return normalized
        else:
            mean = self.rms.mean.item()
            std = np.sqrt(self.rms.var.item() + 1e-8)
            normalized = (rewards - mean) / std
            if self.clip_range is not None:
                normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            return normalized
    
    def state_dict(self):
        return {
            'rms': self.rms.state_dict(),
            'gamma': self.gamma,
            'clip_range': self.clip_range,
            'return': self.return_,
            'episode_rewards': self.episode_rewards
        }
    
    def load_state_dict(self, state_dict):
        self.rms.load_state_dict(state_dict['rms'])
        self.gamma = state_dict['gamma']
        self.clip_range = state_dict['clip_range']
        self.return_ = state_dict.get('return', 0.0)
        self.episode_rewards = state_dict.get('episode_rewards', [])

class DDPGAgent(OffPolicyAgent):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer],
                 action_bound: tuple,
                 actor_hidden_dims: tuple = (256, 256, 256),
                 actor_activation_fn: type[nn.Module] = nn.ReLU,
                 actor_normalizer: BaseNormalizer = None,
                 critic_hidden_dims: tuple = (256, 256, 256),
                 critic_activation_fn: type[nn.Module] = nn.ReLU,
                 critic_normalizer: BaseNormalizer = None,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 noise: Noise = None,
                 reward_normalization: bool = True,  
                 reward_clip_range: float = 10.0,    
                 model_load_path: str = None,
                 model_save_path: str = None,
                 device="cpu"):
        
        super().__init__(replay_buffer=replay_buffer, gamma=gamma, tau=tau, device=device)

        # === Networks ===
        self.actor = ActorNetwork(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, nn.Tanh)
        self.critic = CriticNetwork(obs_dim + act_dim, critic_hidden_dims, critic_activation_fn, nn.Identity)
        self.target_actor = ActorNetwork(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, nn.Tanh)
        self.target_critic = CriticNetwork(obs_dim + act_dim, critic_hidden_dims, critic_activation_fn, nn.Identity)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Normalizers
        if actor_normalizer is None:
            self.actor_normalizer = MinMaxNormalizer(min_val=action_bound[0],
                                                     max_val=action_bound[1], 
                                                     device=device)
        else:
            self.actor_normalizer = actor_normalizer

        if critic_normalizer is None:
            self.critic_normalizer = EmpiricalNormalizer(shape= (obs_dim + act_dim,), device=device)
        else:
            self.critic_normalizer = critic_normalizer
            
            
        self.reward_normalization = reward_normalization
        if self.reward_normalization:
            self.reward_normalizer = RewardNormalizer(
                gamma=gamma, 
                clip_range=reward_clip_range, 
                device=device
            )
            print(f"[✅] Reward normalization enabled (clip_range={reward_clip_range})")
        else:
            self.reward_normalizer = None
            print("[ℹ️] Reward normalization disabled")
            
        # Noise process
        self.noise = noise if noise is not None else OUNoise(size=act_dim, device=device)
            
        # The paths for saving/loading models
        self.load_path = model_load_path
        self.save_path = model_save_path

        # Action bounds
        self.action_bound = action_bound
        
        self.to(device)
        
        # Print model summary
        self.print_model_summary()

    def to(self, device: str = None):
        device = torch.device(device) if device is not None else self.device
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        
    def act(self, obs: torch.Tensor, noise_scale=0.1, explore: bool = True) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(obs)
        if explore:
            action = action + noise_scale * self.noise(action.shape)
        action_clipped = torch.clamp(action, -1.0, 1.0)
        
        # Return denormalized action
        return self.actor_normalizer.denormalize(action_clipped)

    def update(self, batch_size) -> tuple[torch.Tensor, torch.Tensor]:
        """DDPG update step."""
        
        t_batch = self.replay_buffer.sample(batch_size)
        obs, act, rew, next_obs, done = t_batch.obs, t_batch.action, t_batch.reward, t_batch.next_obs, t_batch.done

        # Normalize rewards if normalizer is set
        if self.reward_normalization and self.reward_normalizer is not None:
            rew = self.reward_normalizer.normalize(rew)
            print(f"[DDPG Update] Normalized Rewards: mean={rew.mean().item():.4f}, std={rew.std().item():.4f}")

        with torch.no_grad():
            target_act = self.target_actor(next_obs)
            target_q = self.target_critic(self._critic_input(next_obs, target_act))
            target_val = rew + self.gamma * (1 - done) * target_q

        q = self.critic(self._critic_input(obs, act))
        critic_loss = ((q - target_val) ** 2).mean()

        # actor_loss = -self.critic(self._critic_input(obs, self.actor(obs))).mean()

        # Critic network update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor network update
        actor_actions = self.actor(obs)
        q_values = self.critic(self._critic_input(obs, actor_actions))
        actor_loss = -q_values.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.target_critic, self.critic, self.tau)
        self._soft_update(self.target_actor, self.actor, self.tau)
        
        print(f"[DDPG Update] Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        return actor_loss, critic_loss
        
        
    def save_models(self, path: str):
        """
        Save model parameters and optimizer states.
        Args:
            path (str): Directory path to save model files.
        """
        os.makedirs(path, exist_ok=True)
        try:
            torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
            torch.save(self.target_actor.state_dict(), os.path.join(path, "target_actor.pt"))
            torch.save(self.target_critic.state_dict(), os.path.join(path, "target_critic.pt"))

            torch.save(self.actor_optimizer.state_dict(), os.path.join(path, "actor_optim.pt"))
            torch.save(self.critic_optimizer.state_dict(), os.path.join(path, "critic_optim.pt"))

            # optional: save normalizers
            if hasattr(self, "actor_normalizer"):
                torch.save(self.actor_normalizer.state_dict(), os.path.join(path, "actor_norm.pt"))
            if hasattr(self, "critic_normalizer"):
                torch.save(self.critic_normalizer.state_dict(), os.path.join(path, "critic_norm.pt"))

            print(f"[✅] Models saved successfully to '{path}'")
        except Exception as e:
            warnings.warn(f"Failed to save models: {e}")

    def load_models(self, path: str, strict: bool = True):
        """
        Load model and optimizer parameters.
        Args:
            path (str): Directory where models were saved.
            strict (bool): Whether to enforce strict loading.
        """
        try:
            self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device), strict=strict)
            self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device), strict=strict)
            self.target_actor.load_state_dict(torch.load(os.path.join(path, "target_actor.pt"), map_location=self.device), strict=strict)
            self.target_critic.load_state_dict(torch.load(os.path.join(path, "target_critic.pt"), map_location=self.device), strict=strict)

            # load optimizer states if exist
            actor_optim_path = os.path.join(path, "actor_optim.pt")
            critic_optim_path = os.path.join(path, "critic_optim.pt")
            if os.path.exists(actor_optim_path):
                self.actor_optimizer.load_state_dict(torch.load(actor_optim_path, map_location=self.device))
            if os.path.exists(critic_optim_path):
                self.critic_optimizer.load_state_dict(torch.load(critic_optim_path, map_location=self.device))

            # optional: load normalizers
            actor_norm_path = os.path.join(path, "actor_norm.pt")
            critic_norm_path = os.path.join(path, "critic_norm.pt")
            if hasattr(self, "actor_normalizer") and os.path.exists(actor_norm_path):
                self.actor_normalizer.load_state_dict(torch.load(actor_norm_path))
            if hasattr(self, "critic_normalizer") and os.path.exists(critic_norm_path):
                self.critic_normalizer.load_state_dict(torch.load(critic_norm_path))

            print(f"[✅] Models loaded successfully from '{path}'")
        except Exception as e:
            warnings.warn(f"⚠️ Failed to load models: {e}")
            
    def print_model_summary(self):
        print(f"[DDPG] initialized on {self.device} (γ={self.gamma}, τ={self.tau})")
        print("\n ================ Actor Network ================ \n")
        print(self.actor)
        print(f"Total params: {sum(p.numel() for p in self.actor.parameters() if p.requires_grad):,}")

        print("\n ================ Critic Network ================ \n")
        print(self.critic)
        print(f"Total params: {sum(p.numel() for p in self.critic.parameters() if p.requires_grad):,}")


    def _critic_input(self, obs, act) -> torch.Tensor:
        """Prepare critic input by concatenating observation and action."""
        return torch.cat((obs, act), dim=-1)
        
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    obs_dim = 12
    act_dim = 6
    device = "cuda:0"
    action_bound = (
        torch.tensor([-10., -10., -10., -20., -20., -20.], device=device),
        torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=device)
    )
    
    replay_buffer = SoloReplayBuffer(buffer_size=10000, 
                                     prefill_size=1000,
                                     obs_dim=obs_dim,
                                     action_dim=act_dim, 
                                     device=device)
    
    ddpg_agent = DDPGAgent(obs_dim=obs_dim,
                           act_dim=act_dim,
                           replay_buffer=replay_buffer,
                           action_bound=action_bound,
                           device=device)
    for _ in range(1500):
        transition = TransitionBatch(
            obs=torch.randn(1, obs_dim, device=device),
            action=torch.randn(1, act_dim, device=device),
            reward=torch.randn(1, 1, device=device),
            next_obs=torch.randn(1, obs_dim, device=device),
            done=torch.zeros(1, 1, device=device)
        )
        replay_buffer.add_transition(transition)    

    sample_obs = torch.randn(4, obs_dim, device=device)
    actions = ddpg_agent.act(sample_obs)
    
    actor_loss, critic_loss = ddpg_agent.update(batch_size=64)
    
    print("Sample Actions from DDPG Agent:", actions)
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")