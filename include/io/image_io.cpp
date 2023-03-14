#include "io/image_io.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace mc_vins{
namespace io {

ImageIO::ImageIO(std::string& image_path_in, std::string& image_path_out)
        : image_path_in_(image_path_in), image_path_out_(image_path_out){}

void ImageIO::read_once(cv::Mat& image, const std::string& image_name) {
    image = cv::imread(image_path_in_ + "/" + image_name);
}

void ImageIO::write_once(cv::Mat& image, const std::string& image_name) {
    cv::imwrite(image_path_out_ + "/" + image_name, image);

}

void ImageIO::undistortion(const cv::Mat& image_in, cv::Mat& image_out, std::vector<double> intrinsic, std::vector<double> distortion) {
    int rows = image_in.rows;
    int cols = image_in.cols;
    double fx = intrinsic[0], fy = intrinsic[1], cx = intrinsic[2], cy = intrinsic[3];
    double k1 = distortion[0], k2 = distortion[1], p1 = distortion[2], p2 = distortion[3];

    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            double u_distortion, v_distortion;
            double x1, y1, x2, y2;
            x1 = (u - cx) / fx;
            y1 = (v - cy) / fy;
            double r2 = x1 *x1 + y1 * y1;
            x2 = x1 * (1 + k1 * r2 + k2 * pow(r2, 2)) + 2 * p1 * x1 * y1 + p2 *(r2 + 2 * x1 * x1);
            y2 = y1 * (1 + k1 * r2 + k2 * pow(r2, 2)) + p1 * (r2 + 2 * y1 * y1) + 2 * p2 * x1 * y1;
            u_distortion = fx * x2 + cx;
            v_distortion = fy * y2 + cy;
            if (u_distortion >= 0 && v_distortion >= 0 && u_distortion < cols && v_distortion < rows) {
                // image_out.at<>
            }
        }
    }



}

}
};
