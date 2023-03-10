#pragma once

#include <opencv2/opencv.hpp>
#include "torch/script.h"
#include "features/feature_data.hpp"

namespace mc_vins
{
    class FeatureExtraction
    {
    public:
        struct Options
        {
            std::string superpoint_path;
        };
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FeatureExtraction(const Options &options);
        FeatureData::Ptr extract_features_superpoint(const cv::Mat &img_mat);

    private:
        Options options_;
        torch::jit::script::Module net_superpoint_;
    };
}
