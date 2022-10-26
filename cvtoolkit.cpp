#include "cvtoolkit.h"
#include<opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
std::vector<cv::Scalar> cvt::BASE_COLORMAP = {
    cv::Scalar(128, 0, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(128, 128, 0),
    cv::Scalar(0, 0, 128),
    cv::Scalar(128, 0, 128),
    cv::Scalar(0, 128, 128),
    cv::Scalar(128, 128, 128),
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255)};

cv::Mat cvt::resize(const cv::Mat& img,int width,int height,bool keep_ratio,int pad_value,float* r)
{
    int w = img.cols;
    int h = img.rows;
    int dst_w = 0;
    int dst_h = 0;
    int px = 0;
    int py = 0;
    float _r = 1.0;

    if(!keep_ratio) {
        dst_w = width;
        dst_h = height;
    } else if(width*h>height*w) {
        dst_h = height;
        dst_w = w*height/h;
        px = width -dst_w;
        _r = float(height)/h;
    } else {
        dst_w = width;
        dst_h = width*h/w;
        py = height-dst_h;
        _r = float(width)/w;
    }
    if(r != nullptr) 
        *r = _r;

    cv::Mat dst;
    resize(img,dst,Size(dst_w,dst_h),0,0,INTER_LINEAR);

    if(keep_ratio) {
        Scalar value(pad_value,pad_value,pad_value);
        cv::Mat res;
        copyMakeBorder( dst, res, 0, py, 0, px, cv::BORDER_CONSTANT, value );
        return res;
    } else {
        return dst;
    }
}
cv::Mat cvt::resize_height(const cv::Mat& input,int size,int align)
{
    const int dst_h = size;
    int dst_w = double(dst_h)*input.cols/input.rows;

    if(align>1) {
        dst_w = ((dst_w+align/2)/align)*align;
    }
    cv::Mat dst;

    resize(input,dst,Size(dst_w,dst_h),0,0,INTER_LINEAR);

    return dst;
}
cv::Mat cvt::resize_width(const cv::Mat& input,int size,int align)
{
    const int dst_w = size;
    int dst_h = double(dst_w)*input.rows/input.cols;

    if(align>1) {
        dst_h = ((dst_h+align/2)/align)*align;
    }
    cv::Mat dst;

    resize(input,dst,Size(dst_w,dst_h),0,0,INTER_LINEAR);

    return dst;
}
cv::Mat cvt::resize_short_side(const cv::Mat& input,int size,int align)
{
    if(input.rows<input.cols) {
        return cvt::resize_height(input,size,align);
    } else {
        return cvt::resize_width(input,size,align);
    }
}
cv::Mat cvt::resize_long_side(const cv::Mat& input,int size,int align)
{
    if(input.rows>input.cols) {
        return cvt::resize_height(input,size,align);
    } else {
        return cvt::resize_width(input,size,align);
    }
}
void cvt::draw_keypoints(cv::Mat& img,const cv::Mat& keypoints,const std::vector<std::pair<int,int>>& joints,int radius,bool show_index)
{
    const auto person_nr = keypoints.size[0];
    auto is_good_node = [&keypoints](int i,int j) {
        auto x = keypoints.at<float>(i,j,0);
        auto y = keypoints.at<float>(i,j,1);
        auto p = keypoints.at<float>(i,j,2);
        if((x<=1)||(y<=1)||(p<=0.05))
            return false;
        return true;
    };

    for(auto i=0; i<person_nr; ++i) {
        auto color = Scalar(rand()%255,rand()%255,rand()%255);
        for(auto j=0; j<17; ++j) {
            if(!is_good_node(i,j))
                continue;
            auto x = keypoints.at<float>(i,j,0);
            auto y = keypoints.at<float>(i,j,1);
            cv::circle(img,cv::Point(x,y),radius,color,cv::FILLED);
            if(show_index) 
                cv::putText(img, std::to_string(j), Point(x,y), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0),2);
        }
    }

    int idx0 = 0;
    int idx1 = 0;

    for(auto i=0; i<person_nr; ++i) {
        for(auto x:joints) {
            std::tie(idx0,idx1) = x;
            if(is_good_node(i,idx0) && is_good_node(i,idx1)) {
                auto x0 = keypoints.at<float>(i,idx0,0);
                auto y0 = keypoints.at<float>(i,idx0,1);
                auto x1 = keypoints.at<float>(i,idx1,0);
                auto y1 = keypoints.at<float>(i,idx1,1);
                cv::line(img,cv::Point(x0,y0),cv::Point(x1,y1),Scalar(255,0,0));
            }
        }
    }
}
cv::Mat cvt::resize_subimg_with_pad(const cv::Mat& input, cv::Rect& bbox, int width,int height,int pad_value)
{
    auto       subimg     = cv::Mat(input,bbox);
    const auto w          = subimg.cols;
    const auto h          = subimg.rows;
    const auto ratio      = float(width)/height;
    int        dst_h      = 0;
    int        dst_w      = 0;
    int        px0        = 0;
    int        px1        = 0;
    int        py0        = 0;
    int        py1        = 0;
    auto       box_center = bbox.tl()+cv::Point(bbox.width/2,bbox.height/2);
    auto       box_w      = bbox.width;
    auto       box_h      = bbox.height;
    cv::Mat dst;
    cv::Scalar value(pad_value,pad_value,pad_value);

    if((w==0)||(h==0)) {
        cv::Mat res(height,width,CV_8UC1,value);
        return res;
    } 
    if(width*h>height*w) {
        dst_h = height;
        dst_w = w*height/h;
        px0 = (width-dst_w)/2;
        px1 = width-dst_w-px0;
        box_w = box_h*ratio;
    } else {
        dst_w = width;
        dst_h = width*h/w;
        py0 = (height-dst_h)/2;
        py1 = height-dst_h-py0;
        box_h = box_w/ratio;
    }

    bbox = cv::Rect(box_center-cv::Point(box_w/2,box_h/2),cv::Size(box_w,box_h));

    cv::Mat res;

    resize(subimg,dst,Size(dst_w,dst_h),0,0,INTER_LINEAR);

    copyMakeBorder( dst, res, py0, py1, px0, px1, cv::BORDER_CONSTANT, value );

    if(res.isContinuous())
        return res;
    else
        return res.clone();
}
void cvt::hwc2chw(const cv::Mat& img, float* data,float mean,float var)
{
    const auto img_width = img.cols;
    const auto img_height = img.rows;
    const auto channel = 3;
    const int size0[] = {img_height,img_width,channel};
    const int size1[] = { channel,img_height,img_width };

    const Mat src_mat(3, size0, CV_8UC1, img.data);
    Mat tmp_mat(3, size1, CV_32FC1, data);
    
    for (auto i = 0; i < img_height; ++i) {
        for (auto j = 0; j < img_width; ++j) {
            for (auto k = 0; k < channel; ++k) {
                tmp_mat.at<float>(k, i, j) = src_mat.at<uint8_t>(i, j, k);
            }
        }
    }
}
#ifdef USE_TORCH
torch::Tensor cvt::normalize(const torch::Tensor& data,const std::vector<float>& mean,const std::vector<float>& std)
{
    at::TensorOptions opt = at::TensorOptions().dtype(torch::ScalarType::Float).device(data.device());
    const torch::Tensor t_mean = torch::from_blob((void*)mean.data(),{mean.size()},opt);
    const torch::Tensor t_std = torch::from_blob((void*)std.data(),{std.size()},opt);

    return (data-t_mean)/t_std;
}
#endif
