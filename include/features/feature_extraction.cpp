#include "features/feature_extraction.hpp"

namespace mc_vins
{
    FeatureExtraction::FeatureExtraction(const Options &options) : options_(options)
    {
        std::cout << "Start load models!" << std::endl;
        net_superpoint_ = torch::jit::load(options_.superpoint_path);
        net_superpoint_.to(at::kCUDA);
        net_superpoint_.eval();
        // torch::manual_seed(1024);
        std::cout << "Finish load models!" << std::endl;
    }

    FeatureData::Ptr FeatureExtraction::extract_features_superpoint(const cv::Mat &img_mat) // imgs: N x BGR(0-255)
    {
        cv::Mat img_gray = img_mat.clone();
        cv::cvtColor(img_gray, img_gray, cv::COLOR_RGB2GRAY);
        img_gray.convertTo(img_gray, CV_32F);
        int W = img_gray.cols, H = img_gray.rows;
        double scale = 1600.0 / std::max<double>(W, H);
        int W_new = std::round(double(W) * scale);
        int H_new = std::round(double(H) * scale);

        int interp = cv::INTER_AREA;
        if (W < W_new || H < H_new)
            interp = cv::INTER_LINEAR;
        cv::resize(img_gray, img_gray, cv::Size(W_new, H_new), 0, 0, interp);
        torch::Tensor image_tensor = torch::from_blob(img_gray.data, {1, 1, img_gray.rows, img_gray.cols}, torch::kFloat).to(at::kCUDA);
        image_tensor = image_tensor / 255.0;

        auto result = net_superpoint_.forward({image_tensor}).toGenericDict();
        torch::Tensor keypoints = result.at("keypoints").toTensorVector()[0];
        // torch::Tensor scores = result.at("scores").toTensorVector()[0];
        torch::Tensor descriptors = result.at("descriptors").toTensorVector()[0];
        keypoints.index({torch::indexing::Slice(), 0}) = (keypoints.index({torch::indexing::Slice(), 0}) + 0.5) / W_new * W - 0.5;
        keypoints.index({torch::indexing::Slice(), 1}) = (keypoints.index({torch::indexing::Slice(), 1}) + 0.5) / H_new * H - 0.5;

        FeatureData::Ptr feature_ptr = std::make_shared<FeatureData>();
        feature_ptr->key_points.reserve(keypoints.sizes()[0]);
        feature_ptr->key_points.clear();
        for (int ii = 0; ii < keypoints.sizes()[0]; ii++)
        {
            feature_ptr->key_points.emplace_back(keypoints[ii][0].item<double>(), keypoints[ii][1].item<double>());
        }
        feature_ptr->descriptors = descriptors; //.to(torch::kCPU);
        return feature_ptr;
    }
}
