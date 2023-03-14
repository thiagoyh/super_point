#pragma once

#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>

namespace mc_vins {
namespace io {
class ImageIO {
public:
    ImageIO(std::string& image_path_in, std::string& image_path_out);
    void read_batch();
    void read_once(cv::Mat& image);
    void read_once(cv::Mat& image, const std::string& image_name);
    void undistortion(const cv::Mat& image_in, cv::Mat& image_out, std::vector<double> intrinsic, std::vector<double> distortion);
    void write_once(cv::Mat& image, const std::string& image_name);

private:
    std::string image_path_in_;
    std::string image_path_out_;
};
}
} // namespace mc_vins::io
