#include "features/feature_extraction.hpp"
#include "features/feature_matching.hpp"
#include "features/feature_tracker.hpp"
#include "io/image_io.hpp"
#include "signal.h"

#include <dirent.h>


namespace termSignal{

static bool keepRunning = false;
void sig_handler(int sig) {
    if (sig == SIGINT) {
        keepRunning = false;
    }
}

void init() {
    keepRunning = true;
    signal(SIGINT, sig_handler);
}

bool ok() {
    return keepRunning;
}

} //namespace TermSignal

namespace {

int fileNameFilter(const struct dirent * cur) {
    std::string str(cur->d_name);
    if (str.find(".jpg") != std::string::npos) {
        return 1;
    }
    return 0;
}

bool endWith(std::string& str, std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

} // namespace

int main() {
    mc_vins::FeatureExtraction::Options options{"/data/workspace/super_point/weight/superpoint.pt"};
    mc_vins::FeatureMatching::Options options_matching{0.8, -1, true};

    std::string path_in("/data/camera/test_undistortion");
    std::string path_out("/data/data_out/tracker_mask");
    mc_vins::io::ImageIO image_io(path_in, path_out);

    termSignal::init();
    struct dirent ** name_list;
    int n = scandir(path_in.c_str(), &name_list, fileNameFilter, alphasort);

    if (n < 0) {
        std::cerr << "memory error!!\n";
    }
    else {
        std::cout << "total number is: " << n << std::endl;
    }
    mc_vins::FeatureData::Ptr last_super_points;
    mc_vins::tracker::FeatureTracker feature_tracker(options, options_matching);
    for (int i = 0; i < n && termSignal::ok(); i++) {
        std::string file_name = name_list[i]->d_name;
        if (file_name[0] == '.') {
            continue;
        }
        std::string suffix = ".jpg";
        if (!endWith(file_name, suffix)) {
            continue;
        }
        std::cout << name_list[i]->d_name << std::endl;
        cv::Mat image;
        image_io.read_once(image, file_name);
        feature_tracker.feature_track(image);

        std::vector<Eigen::Vector2d> res_keypoints = feature_tracker.res_keypoints_;
        std::vector<Eigen::Vector2d> cur_keypoints = feature_tracker.cur_keypoints_;
        std::vector<int> res_id = feature_tracker.ids_res_;
        std::vector<int> track_cout_res = feature_tracker.track_count_res_;
        std::vector<int> matches = feature_tracker.matches;
        std::vector<int> cur_descriptors_id = feature_tracker.cur_descriptors_ids_;
        cv::Mat last_image = feature_tracker.res_img_;
        cv::Mat cur_image = feature_tracker.cur_img_;
        cv::RNG rng(time(0));
        std::vector<cv::Mat> vImgs;
        cv::Mat result;

        vImgs.push_back(last_image);
        vImgs.push_back(cur_image);
        cv::hconcat(vImgs, result);
        for (int i = 0; i < res_keypoints.size(); ++i) {
            if (track_cout_res[i] > 4) {
                int b = rng.uniform(0, 255);
                int g = rng.uniform(0, 255);
                int r = rng.uniform(0, 255);
                // std::cout << "id1: " << res_id[i]
                //       << " id2: " << cur_descriptors_id[matches[i]] << std::endl;
                cv::circle(result, cv::Point2f(res_keypoints[i].x(), res_keypoints[i].y()),
                    3, cv::Scalar(b, g, r), -1);
                cv::circle(result, cv::Point2f(cur_keypoints[matches[i]].x() + 3840, cur_keypoints[matches[i]].y()),
                    3, cv::Scalar(b, g, r), -1);
                cv::line(result, cv::Point2f(res_keypoints[i].x(), res_keypoints[i].y()),
                    cv::Point2f(cur_keypoints[matches[i]].x() + 3840, cur_keypoints[matches[i]].y()),
                    cv::Scalar(b, g, r));
            }
        }

        image_io.write_once(result, file_name);

    }
    return 0;
}
