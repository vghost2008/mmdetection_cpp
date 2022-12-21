#pragma once
#include <opencv.hpp>
#include <vector>


struct DetObj
{
    cv::Rect bbox;
    std::vector<std::vector<cv::Point>> mask_contours;
    int      label = -1;
    float    score = 0.0;
};
using DetObjs = std::vector<DetObj>;
