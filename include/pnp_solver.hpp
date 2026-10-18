#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
// PnP 解算的结果。解算可能失败（比如角点顺序错了），
// 所以用 ok 标记，失败的时候其他成员都是空的，取之前一定要先判断 ok
struct PnpResult
{
    bool ok{false};//解算成功了没有
    cv::Mat rvec;//姿态，3x1 的旋转向量（Rodrigues 形式），单位弧度
    cv::Mat tvec;//位置，3x1，板心在相机坐标系下的坐标，单位米：x右 y下 z前
    double distance{0.0};//到相机的直线距离（= tvec 的模长，比 tvec.z 略大一点）
    double reproj_error{0.0};//重投影误差，单位像素。反映角点顺序和内参对不对，但不反映深度准不准
};
class PnpSolver
{
    public:
        // 传入相机的内参，单位像素。不同的相机（比如测试图片和素材视频）必须各建一个
        PnpSolver(double fx, double fy, double cx, double cy);
    PnpResult solve(const std::vector<cv::Point2f>& img_ordered); //期望的角点顺序是 左上→右上→右下→左下，单位像素
    private:
        cv::Mat K;      // 3x3 内参矩阵
        cv::Mat dist;   // 1x5 畸变系数。这里恒为零，视频帧是先校正过再送进来的
};