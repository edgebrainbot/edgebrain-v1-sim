#  EdgeVelAction.msg

std_msgs/Header header
float64[3] cmd_vel_action   # vx, vy, wz
uint8 action_mode           # 0=STUDENT, 1=TEACHER