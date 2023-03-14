#include "features/feature_tracker.hpp"

namespace mc_vins {
namespace tracker {

int64_t FeatureTracker::n_id_ = 0;

FeatureTracker::FeatureTracker(FeatureExtraction::Options& options_extraction, FeatureMatching::Options& options_matching)
    : feature_extraction_(options_extraction), feature_matcher_(options_matching){}

void FeatureTracker::add_points(std::vector<Eigen::Vector2d>& new_pts) {

}

void FeatureTracker::reduce_points_failure(const std::vector<bool>& track_failure_flag) {
    // 更新 res_keypoints_, track_count_res_, ids_res_
    int j = 0;
    res_keypoints_.resize(last_keypoints_.size());
    ids_res_.resize(last_keypoints_.size());
    track_count_res_.resize(last_keypoints_.size());
    for (size_t i = 0; i < last_keypoints_.size(); ++i) {
        if (!track_failure_flag[i]) {
            res_keypoints_[j] = last_keypoints_[i];
            track_count_res_[j] = track_count_last_[i];
            ids_res_[j] = last_descriptors_ids_[i];
            matches[j] = matches[i];
            j++;
        }
    }
    res_keypoints_.resize(j);
    track_count_res_.resize(j);
    ids_res_.resize(j);
    matches.resize(j);
}


void FeatureTracker::set_mask() {

}

void FeatureTracker::feature_track(const cv::Mat& image) {
    if (cur_img_.empty()) {
        last_img_ = cur_img_ = image;
    }
    else {
        cur_img_ = image;
    }

    // 提取当前帧特征点
    auto super_points = feature_extraction_.extract_features_superpoint(image);
    // 当前帧相比于上一帧新提取的特征点   暂时按照没有匹配的点处理
    std::vector<Eigen::Vector2d> new_points;
    // 如果上一帧有特征点，进行特征点匹配  追踪

    if (last_keypoints_.size() > 0) {
        // 当前的特征点 = 追踪到的 +  新出现的
        // 当前描述子对应哪个id  与全部特征点数组长度相同
        cur_descriptors_ids_.resize(super_points->key_points.size(), -1);
        cur_descriptors_ = super_points->descriptors;
        cur_keypoints_ = super_points->key_points;
        track_count_cur_.resize(cur_keypoints_.size(), 1);
        matches = feature_matcher_.find_match(last_descriptors_, cur_descriptors_);
        std::vector<bool> track_failure(matches.size(), true);
        std::vector<bool> new_point_flag(super_points->key_points.size(), true);
        // 判断哪些当前帧特征点被追踪到 matches[i]对应特征点
        // 将cur_img_能与last_img_匹配的特征点按顺序push到cur_pts
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i] != -1) {
                track_failure[i] = false;
                new_point_flag[matches[i]] = false;

                if (last_descriptors_ids_[i] == -1) {
                    last_descriptors_ids_[i] = n_id_++;
                    cur_descriptors_ids_[matches[i]] = last_descriptors_ids_[i];
                }
                else {
                    cur_descriptors_ids_[matches[i]] = last_descriptors_ids_[i];
                }
                // 更新追踪到的每一个特征点以及他对应的id
                // track_count_last_和last_keypoints_的index都是i
                // 匹配的上的话 第matches[i]个特征点
                track_count_last_[i] += 1;
                track_count_cur_[matches[i]] = track_count_last_[i];
            }
        }

        // 去除上一帧中没有被追踪的点
        // 去除没匹配上的上一帧的点 赋值给res_keypoints_
        reduce_points_failure(track_failure);

        // 新的一帧中新出现的特征点 cur_pts_ cur_ids_ cur_pts_descriptors_
        // cur_descriptors_pts_ track_count_cur_ cur_descriptors_ids_
        // 设置mask的话修剪cur_descriptors_
        for (size_t i = 0; i < super_points->key_points.size(); ++i) {
            if (new_point_flag[i]) {
                // 根据mask修剪cur_keypoint和cur_descriptors
                cur_descriptors_ids_[i] = -1;
                track_count_cur_.push_back(1);
            }
        }
        // 每add一个点 id数组长度也会加1 因此下一帧开始ids_和last_pts_长度依然相等
    }

    // 第一帧直接下来
    else {
        last_descriptors_ = super_points->descriptors;
        last_keypoints_ = super_points->key_points;
        last_descriptors_ids_.resize(last_keypoints_.size(), -1);
        track_count_last_.resize(last_keypoints_.size(), 1);
        res_img_ = last_img_;
        return;
    }
    res_img_ = last_img_;
    last_img_ = cur_img_;
    last_descriptors_ = cur_descriptors_;
    last_keypoints_ = cur_keypoints_;
    res_descriptors_ids_ = last_descriptors_ids_;
    last_descriptors_ids_ = cur_descriptors_ids_;
    track_count_res_ = track_count_last_;
    track_count_last_ = track_count_cur_;
    // TODO: 去畸变
    // undistortion_points();
}



// 特征点去畸变，同时计算特征点速度
void FeatureTracker::undistortion_points() {
    cur_un_pts_.clear();


}

} // namespace tracker
} // namespace tracker



