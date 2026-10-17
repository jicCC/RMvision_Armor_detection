#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection 
{
    int   cls;                      // 类别序号（神经方案=装甲板网络输出；传统方案=数字网络输出）
    float score;                    // 这个类别的置信度
    cv::Rect2f box;                 // 外接矩形，给 NMS 用
    std::vector<cv::Point2f> kpt;   // 4 个角点，原图坐标（PnP要用）
};