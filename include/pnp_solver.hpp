#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct PnpResult
{
    bool ok{false};//解算成功了没有
    cv::Mat rvec;//姿态
    cv::Mat tvec;//位置
    double distance{0.0};//到相机的直线距离
    double reproj_error{0.0};//重投影误差
};

class PnpSolver
{
    public:
        PnpSolver(double fx, double fy, double cx, double cy);

    PnpResult solve(const std::vector<cv::Point2f>& img_ordered); //期望的角点顺序是 左上→右上→右下→左下，单位像素

    private:
        cv::Mat K;
        cv::Mat dist;
};