import time
import torch
import numpy as np
from typing import Tuple

from edge_drl_student.common.memory.buffers.base_buffer import Transition
from edge_drl_student.common.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.common.memory.buffers import SoloReplayBuffer
from edge_drl_student.utils.utils import ActionMode
from .base_sampler import BaseSampler


class ACTRCognitiveSampler(BaseSampler):
    """
    ACTRCognitiveSampler
    ---------------
    A sampler that implements ACT-R cognitive sampling strategy.
    This sampler prioritizes experiences based on a cognitive model that
    considers factors such as frequency, recency, and similarity to current state.
    ---------------
    """

    def __init__(self, 
                 replay_buf: SoloReplayBuffer,
                 obs_weight: float = 0.9,
                 action_weight: float = 0.1,
                 family_criteria: float = 0.15,        # default 0.5
                 family_number: int = 50,             # default 55
                 memory_decay_factor: float = -0.5,   # default -0.5
                 fr_weight: float = 0.0,
                 sm_weight: float = 1.0,
                 critic_weight: float = 1.0, 
                 blending_size: int = 5,  #good: 10, 20
                 device: str = "cpu"):
        
        self.replay_buf = replay_buf
        super().__init__(replay_buf.buffer_size)
        
        self.obs_weight = obs_weight
        self.action_weight = action_weight
        self.family_number = family_number
        self.family_criteria = family_criteria
        self.blending_size = blending_size
        self.memory_decay_factor = memory_decay_factor
        
        self.fr_weight = fr_weight         # frequency + recency weight
        self.sm_weight = sm_weight         # similarity weight
        self.critic_weight = critic_weight # per weight
        
        self.device = device
        
        # for per 
        self.critic_scores = torch.zeros(
            self.replay_buf.buffer_size,
            device=self.device
        )
        self._last_indices = None

        self._last_blending_action = torch.Tensor([0.] * 6).to(device)     # Blended action output


        self._last_time = time.time()

    def fetch(self, plant_state: torch.Tensor, batch_size: int, valid_size: int) -> torch.Tensor:

        
        """
        Uniformly sample indices from [0, valid_size).

        Args:
            plant_state (torch.Tensor): Current state of the plant.
            batch_size (int): Number of samples to fetch.
            valid_size (int): Number of valid transitions currently stored.

        Returns:
            torch.Tensor: Randomly sampled indices.
        """
        if valid_size <= 0:
            raise ValueError("Cannot fetch from an empty buffer.")
        if valid_size > self.replay_buf.size:
            raise ValueError(
                f"valid_size ({valid_size}) cannot exceed buffer_size ({self.replay_buf.size})."
            )

        t1 = time.time()
        # Get indices according to time order
        time_idx = torch.arange(1, self.replay_buf.size + 1, device=self.device)
        # print(f"time_idx: {time_idx}")
        
        # TODO: 10w * 10w -> 1w * 1w (generate vector)
        # self.upper_triangular_matrix = torch.ones(self.replay_buf.size, 
        #                                           self.replay_buf.size,
        #                                           device=self.device)
        self.upper_triangular_vector = torch.ones(self.replay_buf.size, 1, device=self.device)
        
        # obs = dual_buf.stu_buf.obs[:dual_buf.stu_buf.size]
        # actions = dual_buf.stu_buf.actions[:dual_buf.stu_buf.size]
        # rewards = dual_buf.stu_buf.rewards[:dual_buf.stu_buf.size]
        # print(f"obs: {obs}")
        # print(f"actions: {actions}")
        # print(f"rewards: {rewards}")
        
        # origin_obs = obs[:,:].square().sum(dim=1)
        # origin_actions = actions[:,:].square().sum(dim=1)
        # print(f"origin_obs: {origin_obs}")
        # print(f"origin_actions: {origin_actions}")
        
        obs = self.replay_buf.obs[:self.replay_buf.size]
        actions = self.replay_buf.actions[:self.replay_buf.size]
        rewards = self.replay_buf.rewards[:self.replay_buf.size]
        
        squeezed_obs = torch.sqrt(obs[:,:].square().sum(dim=1))
        squeezed_actions = torch.sqrt(actions[:,:].square().sum(dim=1))
        
        # print(f"--------------------- ACT-R Sampling Debug Info ---------------------")
        # print(f"squeezed_obs: {squeezed_obs}")
        # print(f"squeezed_actions: {squeezed_actions}")

        compacted_tuple = self.obs_weight * squeezed_obs + self.action_weight * squeezed_actions
        # print(f"compacted_item: {compacted_tuple}")
        # print(f"rewards.squeeze(): {rewards.squeeze()}")
        # print(f"compacted_item before reward weighting: {compacted_tuple}")
        
        # print(f"rewards before exp weighting: {rewards.squeeze()}")
        
        # Apply reward weighting
        rewards = torch.exp(rewards * 0.1)
        # rewards = torch.softmax(rewards.squeeze(), dim=0)
        
        compacted_tuple *= rewards.squeeze()
        # print(f"rewards after exp weighting: {rewards.squeeze()}")
        # print(f"compacted_item after reward weighting: {compacted_tuple}")
        
        family_ID = (compacted_tuple / self.family_criteria).floor()
        # print(f"family_ID: {family_ID}")
        
        
        for i in range(self.family_number):
            family_checking_i = i * torch.ones(self.replay_buf.size, device=self.device)
            # print(f"family_checking_i: {family_checking_i}")
            family_freezing_i = (family_ID == family_checking_i).to(torch.int)
            # print(f"family_freezing_i: {family_freezing_i}")
            # print(f"family_freezing_i dtype: {family_freezing_i.dtype}")
            # print(f"time_idx dtype: {time_idx.dtype}")
            # print(f"family_freezing_i shape: {family_freezing_i.shape}")
            elementwise_fr_score_i = (time_idx + 1).pow(self.memory_decay_factor) * family_freezing_i  # This is elementwise multiplications. Indx indicate time difference;
            # print(f"elementwise_fr_score_i: {elementwise_fr_score_i}")
            # sum_fr_score_i_origin = torch.matmul(self.upper_triangular_matrix, elementwise_fr_score_i)  # This is matrix-vector multiplication. 
            sum_fr_score_i = elementwise_fr_score_i.sum() * self.upper_triangular_vector.squeeze()  # ********** KEY FIX **********
            
            # print(f"sum_fr_score_i_origin: {sum_fr_score_i_origin}")
            # print(f"sum_fr_score_i before log: {sum_fr_score_i}")
            
            # Do the sum for the score:
            # sum_fr_score_i = torch.matmul(self.upper_triangular_matrix, elementwise_fr_score_i)  # This is matrix-vector multiplication. 

            
            # sum_fr_score_i = torch.log(sum_fr_score_i + 1e-8)
            
            sum_fr_score_i = torch.log(sum_fr_score_i + 1.0)
            
            
            fr_score_i = sum_fr_score_i * family_freezing_i  # This is elementwise multiplications.
            
            if i == 0:
                fr_score = fr_score_i
            else:
                fr_score = fr_score + fr_score_i
                
            # print(f"fr_score after family {i}: {fr_score}")

        # real_state = self.replay_buf.obs[self.replay_buf.size - 1]      # Assume last obs is the current state
        # real_state = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0], device="cuda:0")
        sm_error = obs - plant_state    # Similarity error
        # print(f"obs: {obs}")
        # print(f"sm_error: {sm_error}")
        
        sm_score = (0.1 + torch.sqrt(sm_error[:,:].square().sum(dim=1))).pow(-1)  # Similarity score (the smaller the error, the higher the score)  
        
        # --------------------------------------------------
        # Critic loss score
         # --------------------------------------------------
        critic_score = self.critic_scores[:self.replay_buf.size]

        # normalize
        if critic_score.numel() > 1:
            critic_score = (
                critic_score - critic_score.min()
            ) / (
                critic_score.max() - critic_score.min() + 1e-8
            )

      
        # cog_ranking = fr_score * self.fr_weight + sm_score * self.sm_weight
        
        cog_ranking = fr_score * 0.4 + sm_score * 0.6 
        
        
        
        # include per
        # cog_ranking = fr_score*0.0 + sm_score*0.0 + critic_score*1.0
        # print(f"cog_ranking: {cog_ranking}")
        
        # cog_sum = torch.sum(cog_ranking)
        # print(f"cog_ranking: {cog_ranking}")
        # print(f"cog_sum: {cog_sum}")
        
        topk_values, topk_ids = torch.topk(cog_ranking, batch_size)
        
        # Save sampled replay-buffer indices for per
        self._last_indices = topk_ids.detach().clone()
        
        
        batch_samples = self.replay_buf.gather(topk_ids)
        
        # print(f"topk_values: {topk_values}")
        # print(f"topk_ids: {topk_ids}")

        blending_topk_ids = topk_ids[:self.blending_size]
        blending_topk_values = topk_values[:self.blending_size]  
        
        temperature = torch.tensor(0.5, device=self.device)
        blending_topk_values = torch.softmax(blending_topk_values / temperature, dim=-1)
        # print(f"topk_values after softmax: {blending_topk_values}")
        
        blending_action_pool = actions[blending_topk_ids]
        # print(f"actions: {actions}")
        # print(f"blending_action_pool: {blending_action_pool}")
        
        blending_action = torch.sum(blending_action_pool * blending_topk_values.unsqueeze(-1), dim=0)
        # print(f"blending_action: {blending_action}")
        
        self._last_blending_action = blending_action
        
        
        t2 = time.time()
        # print(f"Sampling time: {t2 - t1} seconds")
        # print(f"batch_samples: {batch_samples}")
        
        # print(f"replay_buf name: {self.replay_buf.buf_name}")
        
        return batch_samples

   
    # out for per
    # def update(self, indices: np.ndarray, **kwargs):
    #     """ Update sampling priorities based on provided criteria."""
    #     pass      
    
    def update(self, indices, critic_loss=None, **kwargs):
        """Update critic score for sampled experiences."""
        if critic_loss is None:
            return
        
        indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device
        )
        
        critic_loss = torch.as_tensor(
            critic_loss,
            dtype=torch.float32,
            device=self.device
            ).detach().reshape(-1)
        self.critic_scores[indices] = critic_loss

    def get_blending_action(self) -> torch.Tensor:
        """Get the last blended action computed during sampling.

        Returns:
            torch.Tensor: The last blended action.
        """
        return self._last_blending_action

