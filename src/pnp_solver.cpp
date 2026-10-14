#include "pnp_solver.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

PnpSolver::PnpSolver(double fx, double fy, double cx, double cy)
{
    K = (cv::Mat_<double>(3, 3) << fx,  0, cx,
                                    0, fy, cy,
                                    0,  0,  1);
    dist = cv::Mat::zeros(1, 5, CV_64F);
    // 图片集的内参标定结果里畸变系数全为 0（仿真相机）
    // 以后处理视频时，这里要改成传入视频那一套：k1=-0.09856, k2=0.11173,
    // p1=-0.0038353, p2=-0.0020421, k3=0
}

PnpResult PnpSolver::solve(const std::vector<cv::Point2f>& img_ordered)
{

    PnpResult solve_result = {};  

    std::vector<cv::Point3f> obj_pts = {
        {-0.0675f, -0.0275f, 0.f},   // 左上
        { 0.0675f, -0.0275f, 0.f},   // 右上
        { 0.0675f,  0.0275f, 0.f},   // 右下
        {-0.0675f,  0.0275f, 0.f},   // 左下
    }; 

    //造两个空的 cv::Mat：rvec、tvec
    cv::Mat rvec;      
    cv::Mat tvec;

    //调用 cv::solvePnP
    bool is_solve = cv::solvePnP(obj_pts, img_ordered, K, dist, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    if(!is_solve)
    {
        std::cout << "PnP解算失败！" << std::endl;
        return solve_result;
    }

    double z = tvec.at<double>(2, 0);

    if(z <= 0)
    {
        std::cout << "PnP解算失败！（原因：深度为负，角点顺序可能有误）" << std::endl;
        return solve_result;
    }

    solve_result.ok = true;
    solve_result.rvec = rvec;
    solve_result.tvec = tvec;
    solve_result.distance = cv::norm(tvec);


    std::vector<cv::Point2f> predict_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K, dist, predict_pts);

    double sum = 0;
    for (size_t i = 0; i < img_ordered.size() && i < predict_pts.size(); ++i)
        sum += cv::norm(predict_pts[i] - img_ordered[i]);
    solve_result.reproj_error = predict_pts.empty() ? 0.0 : (sum / predict_pts.size());
    return solve_result;
}