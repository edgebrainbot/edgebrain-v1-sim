#  EdgeAction.msg

std_msgs/Header header
float64[6] control_action   # linear_accel(3) + angular_accel(3)
uint8 action_mode           # 0=STUDENT, 1=TEACHER