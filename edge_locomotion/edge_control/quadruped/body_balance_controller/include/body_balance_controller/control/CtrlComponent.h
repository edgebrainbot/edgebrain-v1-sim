#pragma once

#include "BalanceCtrl.h"
#include "Estimator.h"
#include "body_balance_controller/gait/WaveGenerator.h"
#include "body_balance_controller/robot_model/QuadrupedModel.hpp"

struct CtrlComponent
{
    std::shared_ptr<QuadrupedModel> robot_model_;
    std::shared_ptr<Estimator> estimator_;
    std::shared_ptr<BalanceCtrl> balance_ctrl_;
    std::shared_ptr<WaveGenerator> wave_generator_;

    CtrlComponent() = default;
};
