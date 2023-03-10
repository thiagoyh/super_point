#include "features/feature_matching.hpp"

namespace mc_vins
{
    FeatureMatching::FeatureMatching(const Options &options) : options_(options)
    {
    }

    std::vector<int> FeatureMatching::find_match(const torch::Tensor &desc1, const torch::Tensor &desc2) const
    {
        // std::cout << "Match feature!!" << std::endl;
        torch::Tensor query_local_feature = desc1.to(torch::kCUDA);
        torch::Tensor db_local_feature = desc2.to(torch::kCUDA);
        torch::Tensor sim = torch::einsum("dn,dm->nm", {query_local_feature, db_local_feature.to(torch::kFloat)});
        torch::Tensor matches0 = find_nn(sim, options_.ratio_threshold, options_.distance_threshold);
        if (options_.do_mutual_check)
        {
            torch::Tensor matches1 = find_nn(sim.transpose(0, 1), options_.ratio_threshold, options_.distance_threshold);
            torch::Tensor inds0 = torch::arange(matches0.size(-1)).to(matches0.device());
            torch::Tensor loop = torch::gather(matches1, -1, torch::where(matches0.gt(-1), matches0, 0));
            torch::Tensor ok = matches0.gt(-1) & inds0.eq(loop);
            matches0 = torch::where(ok, matches0, -1);
        }
        matches0 = matches0.to(torch::kInt).to(torch::kCPU);
        std::vector<int> matches(matches0.data_ptr<int>(), matches0.data_ptr<int>() + matches0.numel());
        return matches;
    }

    torch::Tensor FeatureMatching::find_nn(const torch::Tensor &sim, double ratio_threshold, double distance_threshold)
    {
        auto top2 = torch::topk(sim, 2, -1, true);
        torch::Tensor sim_nn = std::get<0>(top2);
        torch::Tensor ind_nn = std::get<1>(top2);
        torch::Tensor dist_nn = 2 * (1 - sim_nn);
        torch::Tensor mask = torch::ones({ind_nn.size(0)}).to(sim.device()).to(torch::kBool);
        if (ratio_threshold > 0.)
            mask = mask & (dist_nn.index({"...", 0}).le((ratio_threshold * ratio_threshold) * dist_nn.index({"...", 1})));
        if (distance_threshold > 0.)
            mask = mask & (dist_nn.index({"...", 0}).le(distance_threshold * distance_threshold));
        torch::Tensor matches = torch::where(mask, ind_nn.index({"...", 0}), long(-1));
        return matches;
    }

    double FeatureMatching::descriptor_distance(const torch::Tensor &a, const torch::Tensor &b)
    {
        /**
        torch::Tensor a = last_super_points->descriptors.index({"...", i});
        torch::Tensor b = super_points->descriptors.index({"...", matches[i]});

        Eigen::
        Eigen::Matrix<double, 256, 1> desc1, desc2;
        for (int i = 0; i < a.size(0); i++) {
            desc1(i) = a[i].item<double>();
            desc2(i) = b[i].item<double>();
        }
        double distance = desc1.dot(desc2);
        **/

        // Tensor::
        auto distance = a.dot(b);
        // std::cout << "distance: " << distance << std::endl;
        return distance.item<double>();
    }

    std::pair<FeatureData::Ptr, int> FeatureMatching::compute_distinctive_descriptors(const std::vector<std::pair<FeatureData::Ptr, int>> &descs) {
        const size_t N = descs.size();
        std::vector<std::vector<double>> distances(N, std::vector<double>(N , -1));
        for (size_t i = 0; i < N; ++i) {
            distances[i][i] = 0;
            for (size_t j = i +1; j < N; ++j) {
                torch::Tensor descriptor_1, descriptor_2;
                descriptor_1 = descs[i].first->descriptors.index({"...", descs[i].second});
                descriptor_2 = descs[j].first->descriptors.index({"...", descs[j].second});
                double dist_ij = descriptor_distance(descriptor_1, descriptor_2);
                distances[i][j] = dist_ij;
                distances[j][i] = dist_ij;
            }
        }
        // choose the most distinctive descriptor which is closest to the other descriptors in this set
        int best_median = INT_MAX, best_index = 0;
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> distances_i2others(distances[i].begin(), distances[i].begin() + N);
            // 排序可以优化，可以直接找中位数
            std::sort(distances_i2others.begin(), distances_i2others.end());

            // get the median
            int median = distances_i2others[0.5 * (N - 1)];
            if (median < best_median) {
                best_median = median;
                best_index = i;
            }
        }

        std::pair<FeatureData::Ptr, int> distictive_descriptor
                = std::make_pair(descs[best_index].first, descs[best_index].second);

        return distictive_descriptor;
    }
}
