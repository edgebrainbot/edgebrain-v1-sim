#pragma once

#include <memory>
#include "body_balance_controller/utils/math/mathTypes.h"
#include "body_balance_controller/gait/FootstepPlanner.hpp"


class Estimator;
class WaveGenerator;
struct CtrlComponent;

namespace edge::locomotion
{
class GaitGenerator {
public:
    explicit GaitGenerator(CtrlComponent &ctrl_component);

    ~GaitGenerator() = default;

    void setGait(Vec2 vxy_goal_global, double d_yaw_goal, double gait_height);

    void generate(Vec34 &feet_pos, Vec34 &feet_vel);

    void restart();

private:
    Vec3 getFootPos(int i);

    Vec3 getFootVel(int i);

    /**
     * Calculate the position of the foot in the XY plane
     * @param startXY
     * @param endXY
     * @param phase
     * @return
     */
    static double cycloidXYPosition(double startXY, double endXY, double phase);

    /**
     * Calculate the position of the foot in the Z direction
     * @param startZ
     * @param height
     * @param phase
     * @return
     */
    static double cycloidZPosition(double startZ, double height, double phase);

    /**
     * Calculate the velocity of the foot in the XY plane
     * @param startXY
     * @param endXY
     * @param phase
     * @return
     */
    [[nodiscard]] double cycloidXYVelocity(double startXY, double endXY, double phase) const;

    /**
     * Calculate the velocity of the foot in the Z direction
     * @param height
     * @param phase
     * @return
     */
    [[nodiscard]] double cycloidZVelocity(double height, double phase) const;

    std::shared_ptr<WaveGenerator> &wave_generator_;
    std::shared_ptr<Estimator> &estimator_;
    FootstepPlanner footstep_planner_;

    double gait_height_{};
    Vec2 vxy_goal_;
    double d_yaw_goal_{};
    Vec34 start_p_, end_p_, ideal_p_, past_p_;
    bool first_run_;
};

} // namespace edge::locomotion
