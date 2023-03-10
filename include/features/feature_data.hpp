#pragma once

#include <Eigen/Core>
#include "torch/torch.h"

namespace mc_vins
{
    struct FeatureData
    {
        typedef std::shared_ptr<FeatureData> Ptr;
        std::vector<Eigen::Vector2d> key_points;
        torch::Tensor descriptors;
    };
}
