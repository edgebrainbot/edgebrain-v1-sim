#pragma once

#include "BalanceCtrl.h"
#include "Estimator.h"
#include "body_balance_controller/gait/WaveGenerator.h"
#include "body_balance_controller/robot_model/QuadrupedModel.hpp"

struct CtrlComponent
{
    std::shared_ptr<edge::locomotion::QuadrupedModel> robot_model_;
    std::shared_ptr<edge::locomotion::Estimator> estimator_;
    std::shared_ptr<edge::locomotion::BalanceCtrl> balance_ctrl_;
    std::shared_ptr<edge::locomotion::WaveGenerator> wave_generator_;

    CtrlComponent() = default;
};
