#pragma once

#include "features/feature_extraction.hpp"
#include "features/feature_matching.hpp"

namespace mc_vins {
namespace tracker {

class FeatureTracker {
public:
    FeatureTracker(FeatureExtraction::Options& options_extraction, FeatureMatching::Options& options_matching);
    void feature_track(const cv::Mat& image);
    static int64_t n_id_;
private:
    void add_points(std::vector<Eigen::Vector2d>& new_pts);
    void reduce_points_failure(const std::vector<bool>& track_failure_flag);
    void reduce_ids(std::vector<int>& ids, const std::vector<bool>& track_failure_flag);
    void undistortion_points();
    bool updateID(unsigned int i);
    void set_mask();
    FeatureExtraction feature_extraction_;
    FeatureMatching feature_matcher_;
public:
    cv::Mat last_img_, cur_img_, res_img_;
    torch::Tensor last_descriptors_, cur_descriptors_;
    std::vector<Eigen::Vector2d> last_pts_, cur_pts_, res_pts_;
    std::vector<int> last_descriptors_ids_;
    std::vector<int> res_descriptors_ids_;
    std::vector<int> cur_descriptors_ids_;
    std::vector<int> last_pts_descriptors_;
    std::vector<int> res_pts_descriptors_;
    std::vector<int> cur_pts_descriptors_;
    std::vector<int> last_descriptors_pts_; // 描述子对应数组的位置
    std::vector<int> cur_descriptors_pts_;
    std::vector<int> track_count_last_; // 跟last_pts_下标一致 表示对应位置特征点被追踪几次
    std::vector<int> track_count_cur_;
    std::vector<int> track_count_res_;
    std::vector<int> cur_ids_;
    std::vector<Eigen::Vector2d> last_un_pts_, cur_un_pts_;
    std::vector<int> matches;

    double cur_time_;
    double prev_time_;

};

} // namespace tracker
} // namespace mc_vins
