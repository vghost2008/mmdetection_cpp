#pragma once
#include <torch/script.h>
#include <string>
#include <opencv.hpp>
#include "detection_fwd.h"

class FasterRCNN
{
    public:
        FasterRCNN(const std::string& model_path,const cv::Size& input_size,float score_threshold=0.05f);
        /*
         * img: rgb order
         *
         */
        DetObjs forward(const cv::Mat& img);
        std::vector<DetObjs> forward(const std::vector<cv::Mat>& imgs);
    private:
        DetObjs get_results(const torch::Tensor& data,float r=1.0f);
        std::vector<DetObjs> get_results(const torch::Tensor& data,std::vector<float> r);
        torch::Tensor prepare_tensor(const cv::Mat& img,float* r);
    private:
        torch::jit::script::Module module_;
        cv::Size input_size_;
        float score_threshold_ = 0.0f;
};
