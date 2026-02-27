#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

//高层状态topic，其中rt表示实时，lf表示低频
#define TOPIC_HIGHSTATE "rt/sportmodestate"

using namespace unitree::robot;

//获取运动状态的回调函数
void HighStateHandler(const void* message)
{
    unitree_go::msg::dds_::SportModeState_ state = *(unitree_go::msg::dds_::SportModeState_*)message;

    // Print the position of the robot (x, y, z)
    std::cout << "position: "
            << "x: " << state.position()[0] << ", "
            << "y: " << state.position()[1] << ", "
            << "z: " << state.position()[2] << std::endl;

    // Print the velocity of the robot (vx, vy, vz)
    std::cout << "velocity: "
            << "vx: " << state.velocity()[0] << ", "
            << "vy: " << state.velocity()[1] << ", "
            << "vz: " << state.velocity()[2] << std::endl;

    // Print the euler angles of the robot (roll, pitch, yaw)
    std::cout << "euler angles: "
            << "roll: " << state.imu_state().rpy()[0] << ", "
            << "pitch: " << state.imu_state().rpy()[1] << ", "
            << "yaw: " << state.imu_state().rpy()[2] << std::endl;

    // Print the angular velocity of the robot (wx, wy, wz)
    std::cout << "angular velocity: "
            << "wx: " << state.imu_state().gyroscope()[0] << ", "
            << "wy: " << state.imu_state().gyroscope()[1] << ", "
            << "wz: " << state.imu_state().gyroscope()[2] << std::endl;
}


int main()
{
  //初始化sdk接口
  std::string networkInterface = "eno1";//机器人连接的网卡名称
  unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

  std::cout << "Start to subscribe " << TOPIC_HIGHSTATE << std::endl;

  //创建一个Subscriber
  ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> suber(TOPIC_HIGHSTATE);

  //初始化Channel
  suber.InitChannel(HighStateHandler);

  while(1)
  {
    usleep(20000);
  }

  return 0;
}