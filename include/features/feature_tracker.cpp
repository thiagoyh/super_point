#include "features/feature_tracker.hpp"

namespace mc_vins {
namespace tracker {

int64_t FeatureTracker::n_id_ = 0;

FeatureTracker::FeatureTracker(FeatureExtraction::Options& options_extraction, FeatureMatching::Options& options_matching)
    : feature_extraction_(options_extraction), feature_matcher_(options_matching){}

void FeatureTracker::add_points(std::vector<Eigen::Vector2d>& new_pts) {
    for (auto& p : new_pts) {
        cur_pts_.push_back(p);
        cur_ids_.push_back(-1);
        track_count_cur_.push_back(1);
    }
}

void FeatureTracker::reduce_points_failure(const std::vector<bool>& track_failure_flag) {
    // 更新 last_pts_, last_ids_, last_pts_descriptors_
    int j = 0;
    for (size_t i = 0; i < last_pts_.size(); ++i) {
        // 拿到对应描述子
        int desc_position = last_pts_descriptors_[i];
        if (!track_failure_flag[desc_position]) {
            last_pts_[j] = last_pts_[i];
            // last_ids_[j] = last_ids_[i];
            track_count_res_[j] = track_count_res_[i];
            last_pts_descriptors_[j] = desc_position;
            // last_descriptors_pts_[last_pts_descriptors_[j]] =
            j++;
        }
    }

    last_pts_.resize(j);
    last_pts_descriptors_.resize(j);
    track_count_res_.resize(j);
}
void FeatureTracker::reduce_ids(std::vector<int>& ids, const std::vector<bool>& track_failure_flag) {
    int j = 0;
    for (size_t i = 0; i < ids.size(); i++) {
        if (track_failure_flag[i]) {
            ids[j++] = ids[i];
        }
    }
    ids.resize(j);
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

    if (last_pts_.size() > 0) {
        // 当前的特征点 = 追踪到的 +  新出现的
        cur_pts_.clear();
        // 当前的特征点的id    一一对应
        cur_ids_.clear();
        // 当前特征点是对应哪个描述子  与cur_pts_长度相同
        cur_pts_descriptors_.clear();
        track_count_cur_.clear();
        // 当前描述子对应哪个id  与全部特征点数组长度相同
        cur_descriptors_ids_.resize(super_points->key_points.size(), -1);
        cur_descriptors_pts_.resize(super_points->key_points.size(), -1);
        cur_descriptors_ = super_points->descriptors;
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
                cur_pts_.push_back(super_points->key_points[matches[i]]);
                cur_ids_.push_back(cur_descriptors_ids_[matches[i]]);
                cur_pts_descriptors_.push_back(matches[i]);
                cur_descriptors_pts_[matches[i]] = cur_pts_.size() - 1;
                // last_pts_数组中的位置
                int pts_position = last_descriptors_pts_[i];
                track_count_last_[pts_position] += 1;
                track_count_cur_.push_back(-1);
                track_count_cur_[cur_pts_.size() - 1] = track_count_last_[pts_position];
            }
        }
        track_count_res_ = track_count_last_;
        // 去除上一帧中没有被追踪的点
        // last_pts_, last_ids_, last_pts_descriptors_, track_count_
        // 这一步做完 cur_pts和last_pts长度一致
        reduce_points_failure(track_failure);
        res_pts_ = last_pts_;

        // 新的一帧中新出现的特征点 cur_pts_ cur_ids_ cur_pts_descriptors_
        // cur_descriptors_pts_ track_count_cur_ cur_descriptors_ids_
        // 这个地方set_mask
        for (size_t i = 0; i < super_points->key_points.size(); ++i) {
            if (new_point_flag[i]) {
                cur_pts_.push_back(super_points->key_points[i]);
                cur_ids_.push_back(-1);
                cur_pts_descriptors_.push_back(i);
                cur_descriptors_pts_[i] = cur_pts_.size() - 1;
                cur_descriptors_ids_[i] = -1;
                track_count_cur_.push_back(1);
            }
        }
        // 每add一个点 id数组长度也会加1 因此下一帧开始ids_和last_pts_长度依然相等
    }

    // 第一帧直接下来
    else {
        cur_pts_.reserve(super_points->key_points.size());
        cur_ids_.reserve(super_points->key_points.size());
        track_count_cur_.reserve(super_points->key_points.size());
        add_points(super_points->key_points);
        // 记下每一个描述子对应的id
        cur_descriptors_ids_.resize(super_points->key_points.size(), -1);
        // 记录每一个选中的特征点对应的描述子
        cur_pts_descriptors_.reserve(super_points->key_points.size());
        cur_descriptors_pts_.reserve(super_points->key_points.size());
        for (size_t i = 0; i < super_points->key_points.size(); ++i) {
            cur_pts_descriptors_.push_back(i);
            cur_descriptors_pts_.push_back(i);
        }
        cur_descriptors_ = super_points->descriptors;
    }

    res_img_ = last_img_;
    last_img_ = cur_img_;
    last_descriptors_ = cur_descriptors_;
    last_pts_ = cur_pts_;
    // last_ids_ = cur_ids_;
    res_pts_descriptors_ = last_pts_descriptors_;
    last_pts_descriptors_ = cur_pts_descriptors_;

    last_descriptors_pts_ = cur_descriptors_pts_;

    res_descriptors_ids_ = last_descriptors_ids_;
    last_descriptors_ids_ = cur_descriptors_ids_;

    track_count_last_ = track_count_cur_;
    // TODO: 去畸变
    // undistortion_points();
}



// 特征点去畸变，同时计算特征点速度
void FeatureTracker::undistortion_points() {
    cur_un_pts_.clear();


    for (size_t i = 0; i < cur_pts_.size(); i++) {

    }

}

} // namespace tracker
} // namespace tracker



