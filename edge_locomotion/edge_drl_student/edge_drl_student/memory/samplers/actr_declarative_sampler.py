import torch
import numpy as np

from edge_drl_student.memory.buffers.base_buffer import TransitionData
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.utils.utils import ActionMode
from .base_sampler import BaseSampler


class ACTRDeclarativeSampler(BaseSampler):
    """
    ACTRDeclarativeSampler
    ---------------
    A simple sampler that draws indices uniformly at random
    from the valid portion of the replay buffer.
    """

    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)

    def fetch(self, batch_size: int, valid_size: int) -> np.ndarray:
        """
        Uniformly sample indices from [0, valid_size).

        Args:
            batch_size (int): Number of samples to fetch.
            valid_size (int): Number of valid transitions currently stored.

        Returns:
            np.ndarray[int]: Randomly sampled indices.
        """
        if valid_size <= 0:
            raise ValueError("Cannot fetch from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) cannot exceed buffer_size ({self.buffer_size})."
            )

        # indices = np.random.randint(0, valid_size, size=batch_size)
        
        # time_idx = torch.arange(1, batch_size + 1)
        # time_index = np.arange(1, (batch_size+1)); 

        # family_ID = int(v(s)/family_criteria);
        # upper_triangular_matrix = np.triu(np.ones((reply_buffer_size, reply_buffer_size))); Create upper triangular matrix with ones.
        # If instances in the same family are given the same score, choose 
        # upper_triangular_matrix = np.ones((reply_buffer_size, reply_buffer_size)); 

        # for i = 0 : (family_number-1); 
        # family_checking_i = i *  ones(1,replay_buffer_size);
        # family_freezing_i = (family_group == family_checking_i).astype(int); 
        # elementwise_fr_score_i = (indx + 1)^{-0.5} * family_freezing_i;  This is elementwise multiplications. Indx indicate time difference;
        # sum_fr_score_i = upper_triangular_matrix * elementwise_fr_score_i;  This is matrix-vector multiplication. 
        # sum_fr_score_i = log(sum_fr_score_i)
        # fr_score_i = sum_fr_score_i * family_freezing_i;  This is elementwise multiplications.
        # fr_score_i = fr_score_i + fr_score_i-1;
        # end for 
        
        # return indices

    def update(self, indices: np.ndarray, **kwargs):
        """Uniform sampler has no internal state to update."""
        pass


if __name__ == "__main__":
    # ======================================================
    # 1. create buffer
    # ======================================================
    dual_buf = DualReplayBuffer(
        student_buf_size=10,
        student_prefill_size=3,
        teacher_buf_size=10,
        teacher_prefill_size=3,
        obs_dim=12,
        action_dim=6,
        device="cuda:0"
    )

    # ======================================================
    # 2. write in transition
    # ======================================================
    for i in range(24):
        t = TransitionData(
            obs=torch.tensor([i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10, i+11], dtype=torch.float32),
            action=torch.tensor([i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5, i * 0.6], dtype=torch.float32),
            reward=torch.tensor([i * 1.0], dtype=torch.float32),
            next_obs=torch.tensor([i+0.1, i+1.1, i+2.1, i+3.1, i+4.1, i+5.1, i+6.1, i+7.1, i+8.1, i+9.1, i+10.1, i+11.1], dtype=torch.float32),
            done=torch.tensor([i % 2 == 0], dtype=torch.float32)
        )
        dual_buf.add_transition(t, action_mode=ActionMode.STUDENT if i % 2 == 0 else ActionMode.TEACHER)
        
    print(f"Student Buffer Size: {dual_buf.stu_buf_size}")
    print(f"Teacher Buffer Size: {dual_buf.tea_buf_size}")
    
    
    time_idx = torch.arange(1, dual_buf.stu_buf.size+1, device="cuda:0")
    print(f"time_idx: {time_idx}")
    
    
    obs = dual_buf.stu_buf.obs[:dual_buf.stu_buf.size]
    actions = dual_buf.stu_buf.actions[:dual_buf.stu_buf.size]
    rewards = dual_buf.stu_buf.rewards[:dual_buf.stu_buf.size]
    print(f"obs: {obs}")
    print(f"actions: {actions}")
    print(f"rewards: {rewards}")
    
    squeezed_obs = obs[:,:].square().sum(dim=1)
    squeezed_actions = actions[:,:].square().sum(dim=1)
    print(f"squeezed_obs: {squeezed_obs}")
    print(f"squeezed_actions: {squeezed_actions}")

    obs_weight = 0.1
    actions_weight = 0.9
    
    compacted_tuple = obs_weight * squeezed_obs + actions_weight * squeezed_actions
    print(f"compacted_item: {compacted_tuple}")
    
    family_criteria = torch.tensor(100.0, device="cuda:0")
    family_ID = (compacted_tuple/family_criteria).floor()
    
    print(f"family_ID: {family_ID}")
    
    upper_triangular_matrix = torch.ones(dual_buf.stu_buf.size, dual_buf.stu_buf.size, device="cuda:0")
    print(f"upper_triangular_matrix: {upper_triangular_matrix}")
    
    family_number = 15
    memory_decay_factor = -0.5
    
    for i in range(family_number):
        family_checking_i = i * torch.ones(dual_buf.stu_buf.size, device="cuda:0")
        # print(f"family_checking_i: {family_checking_i}")
        family_freezing_i = (family_ID == family_checking_i).to(torch.int8)
        # print(f"family_freezing_i: {family_freezing_i}")
        elementwise_fr_score_i = (time_idx + 1).pow(memory_decay_factor) * family_freezing_i  # This is elementwise multiplications. Indx indicate time difference;
        # print(f"elementwise_fr_score_i: {elementwise_fr_score_i}")
        sum_fr_score_i = torch.matmul(upper_triangular_matrix, elementwise_fr_score_i)  # This is matrix-vector multiplication. 
        
        sum_fr_score_i = torch.log(sum_fr_score_i + 1e-8)
        
        fr_score_i = sum_fr_score_i * family_freezing_i  # This is elementwise multiplications.
        
        
        
        if i == 0:
            fr_score = fr_score_i
        else:
            fr_score = fr_score + fr_score_i
            
        print(f"fr_score after family {i}: {fr_score}")

    
    # test = obs[0,:] @ state_mat @ obs[:,0]
    # print(f"test: {test}")
    # # test = obs * state_mat * obs
    # V_state = torch.matmul(torch.matmul(obs, state_mat), obs)
    # print(f"V_state: {V_state}")
    
    # q_values = torch.tensor([1.0, 2.0, 0.5])
    
    # Boltzmann distribution
    print(f"fr_score: {fr_score}")
    temperature = torch.tensor(0.5, device="cuda:0")
    probs = torch.softmax(fr_score / temperature, dim=-1)
    print(f"probs: {probs}")
    
    
    sampled_batch = dual_buf.sample(batch_size=5)
    print(f"Sampled Batch: {sampled_batch}")
    
    dual_buf.stu_buf.dump(path="./stu_buffer_dump.txt", format='csv')
    dual_buf.tea_buf.dump(path="./tea_buffer_dump.txt", format='csv')