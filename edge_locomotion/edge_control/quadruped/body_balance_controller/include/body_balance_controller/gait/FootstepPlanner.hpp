
#ifndef FOOTSTEPPLANNER_HPP
#define FOOTSTEPPLANNER_HPP
#include <memory>
#include "body_balance_controller/utils/math/mathTypes.h"


class Estimator;
struct CtrlComponent;
class QuadrupedModel;

namespace edge::locomotion
{

class FootstepPlanner {
public:
    explicit FootstepPlanner(CtrlComponent &ctrl_component);

    ~FootstepPlanner() = default;

    Vec3 calcFootPos(int index, Vec2 vxy_goal_global, double d_yaw_global, double phase);

private:
    CtrlComponent &ctrl_component_;
    std::shared_ptr<QuadrupedModel> &robot_model_;
    std::shared_ptr<Estimator> &estimator_;

    Vec4 feet_radius_, feet_init_angle_;

    double k_x_, k_y_, k_yaw_;
    double t_stance_{}, t_swing_{};
};

} // namespace edge::locomotion

#endif //FOOTSTEPPLANNER_HPP
