# Transition Experience for replay buffer

std_msgs/Header header
float64[] obs
float64[] action
Reward reward
float64[] next_obs
bool done
uint8 action_mode           # 0=STUDENT, 1=TEACHER
