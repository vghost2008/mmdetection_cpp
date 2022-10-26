#pragma once
#include <torch/script.h>
#include <string>
#include <opencv.hpp>
#include "detection_fwd.h"

class FasterRCNN
{
    public:
        FasterRCNN(const std::string& model_path,const cv::Size& input_size,float score_threshold=0.05f);
        DetObjs forward(const cv::Mat& img);
    private:
        DetObjs get_results(const torch::Tensor& data,float r=1.0f);
    private:
        torch::jit::script::Module module_;
        cv::Size input_size_;
        float score_threshold_ = 0.0f;
};
