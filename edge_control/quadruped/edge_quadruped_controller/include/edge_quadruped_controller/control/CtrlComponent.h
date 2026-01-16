#pragma once

#include "edge_quadruped_controller/control/BalanceCtrl.h"
#include "edge_quadruped_controller/gait/PhaseScheduler.h"
#include "edge_quadruped_controller/estimator/Estimator.hpp"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"

namespace edge::control::quadruped
{

struct CtrlComponent
{
    std::shared_ptr<QuadrupedModel> robot_model_;
    std::shared_ptr<Estimator> estimator_;
    std::shared_ptr<BalanceCtrl> balance_ctrl_;
    std::shared_ptr<PhaseScheduler> phase_scheduler_;

    CtrlComponent() = default;
};

} // namespace edge::control::quadruped
