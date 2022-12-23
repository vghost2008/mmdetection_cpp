#pragma once
#include <torch/script.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "detection_fwd.h"

class FasterRCNN
{
    public:
        FasterRCNN(const std::string& model_path,
                   const cv::Size& input_size,
                   float score_threshold=0.05f,
                   const std::vector<float>& mean={123.675, 116.28, 103.53}, 
                   const std::vector<float>& std={58.395, 57.12, 57.375});
        virtual ~FasterRCNN();
        /*
         * img: rgb order
         *
         */
        DetObjs forward(const cv::Mat& img);
        std::vector<DetObjs> forward(const std::vector<cv::Mat>& imgs);
    public:
        inline float score_threshold()const {
            return score_threshold_;
        }
        inline void set_score_threshold(float v) {
            score_threshold_ = v;
        }
    protected:
        virtual DetObjs get_results(const std::vector<torch::Tensor>& data,float r=1.0f);
        virtual std::vector<DetObjs> get_results(const std::vector<torch::Tensor>& data,std::vector<float> r);
        torch::Tensor prepare_tensor(const cv::Mat& img,float* r);
    protected:
        torch::jit::script::Module module_;
        cv::Size input_size_;
        float score_threshold_ = 0.0f;
        std::vector<float> mean_;
        std::vector<float> std_;
};
