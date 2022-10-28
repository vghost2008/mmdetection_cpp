#pragma once
#include <opencv.hpp>
#include <vector>
#include <utility>
#include <string>
#ifdef USE_TORCH
#include <torch/script.h>
#endif

namespace cvt
{
    extern std::vector<cv::Scalar> BASE_COLORMAP;
    cv::Mat resize(const cv::Mat& input,int width,int height,bool keep_ratio=true,int pad_value=127,float* r=nullptr/*new_size/old_size*/);
    cv::Mat resize_short_side(const cv::Mat& input,int size,int align=1);
    cv::Mat resize_long_side(const cv::Mat& input,int size,int align=1);
    cv::Mat resize_height(const cv::Mat& input,int size,int align=1);
    cv::Mat resize_width(const cv::Mat& input,int size,int align=1);
    cv::Mat resize_subimg_with_pad(const cv::Mat& input, cv::Rect& bbox, int width,int height,int pad_value=127);
    void hwc2chw(const cv::Mat& img, float* data,float mean=0.0,float var=1.0);
    template<typename T>
        cv::Mat resize_subimg_with_pad(const cv::Mat& input, cv::Rect_<T>& bbox, int width,int height,int pad_value=127) {
            cv::Rect r = bbox;
            auto res = resize_subimg_with_pad(input,r,width,height,pad_value);
            bbox = r;
            return res;
        }
    /*
     * img: [H,W,C]
     * keypoints: [person_nr,keypoints_nr,3+x]
     */
    void draw_keypoints(cv::Mat& img,const cv::Mat& keypoints,const std::vector<std::pair<int,int>>& joints,int radius=3,bool show_index=false);
    template<typename DetObjs>
        void draw_bboxes_dets(cv::Mat& img,const DetObjs& objs) {
            for(auto& obj:objs) {
                const auto &rect     = obj.bbox;
                int         class_id = obj.label;
                float       score    = obj.score;
                const auto  color    = BASE_COLORMAP[class_id%BASE_COLORMAP.size()];

                cv::rectangle(img, rect, color, 2);
                cv::putText(img, std::to_string(class_id)+"_"+std::to_string(int(score*100)), cv::Point(rect.x, rect.y), 
                            cv::FONT_HERSHEY_DUPLEX, 0.8f, cv::Scalar(0,0,255));
                            //cv::FONT_HERSHEY_DUPLEX, 0.8f, color);
                            //cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
            }
        }

    inline void default_log_func(const std::string& v)
    {
        std::cout<<v<<std::endl;
    }
    class WTimeThis {
        public:
            WTimeThis(const std::string& name,std::function<void(const std::string&)> func=default_log_func,bool autolog=true)
                :name_(name),func_(func),t_(std::chrono::steady_clock::now()),autolog_(autolog)
            {
                func_(name_);
            }
            ~WTimeThis() {
                if(autolog_)
                    log();
            }
            inline void log_and_reset(const std::string& name) {
                log(name);
                reset();
            }
            inline void log_and_reset() {
                log_and_reset(name_);
            }
            inline void reset() {
                t_ = std::chrono::steady_clock::now();
            }
            inline int time_duration()const {
                return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_).count();
            }
            inline void log()const {
                log(name_);
            }
            inline void log(const std::string& name)const {
                std::stringstream ss;
                ss<<name<<":"<<time_duration()<<" milliseconds, fps="<<1000.0/time_duration();
                func_(ss.str());
            }
        private:
            const std::string name_;
            std::function<void(const std::string&)> func_;
            std::chrono::steady_clock::time_point t_;
            bool autolog_ = false;
    };
#ifdef USE_TORCH
    torch::Tensor normalize(const torch::Tensor& data,const std::vector<float>& mean,const std::vector<float>& std);
#endif
}
