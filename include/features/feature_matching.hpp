#pragma once

#include "features/feature_data.hpp"

namespace mc_vins
{
    class FeatureMatching
    {
    public:
        struct Options
        {
            double ratio_threshold = 0.8;
            double distance_threshold = -1.;
            bool do_mutual_check = true;
        };
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FeatureMatching(const Options &options);
        std::vector<int> find_match(const torch::Tensor &desc1, const torch::Tensor &desc2) const;
        static torch::Tensor find_nn(const torch::Tensor &sim, double ratio_threshold = -1.0, double distance_threshold = -1.0);
        static double descriptor_distance(const torch::Tensor &a, const torch::Tensor &b);//TODO
        static std::pair<FeatureData::Ptr, int> compute_distinctive_descriptors(const std::vector<std::pair<FeatureData::Ptr, int>> &descs);//TODO

    private:
        Options options_;
    };
}
