#include "pnp_solver.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace {

// 装甲板物理尺寸（小装甲板 135mm × 55mm）。物点以板心为原点，
// x 向右、y 向下、z 垂直板面向外，和 OpenCV 的相机坐标系一致。
constexpr float ARMOR_HALF_W   = 0.0675f;   // 135/2
constexpr float ARMOR_HALF_H   = 0.0275f;   // 55/2
constexpr float ARMOR_PLANE_Z  = 0.0f;      // 平面靶标，z 恒为 0
constexpr int   DIST_COEFF_NUM = 5;         // 畸变系数个数：k1,k2,p1,p2,k3

} // namespace

PnpSolver::PnpSolver(double fx, double fy, double cx, double cy)
{
    // 内参矩阵，OpenCV 的标准写法
    K = (cv::Mat_<double>(3, 3) << fx,  0, cx,
                                    0, fy, cy,
                                    0,  0,  1);

    // 这里始终用零畸变，不去传真实的畸变系数。
    // 因为两条路都是"无畸变"的输入：测试图片本身没畸变；视频帧在 main 里
    // 已经先用 cv::undistort 校正过了。这样检测和解算用的就是同一种图像，
    // 不会出现"在畸变图上检测、用理想模型解算"的不一致。
    // （视频那一套系数是 k1=-0.0986, k2=0.1117, p1=-0.0038, p2=-0.0020，在 main.cpp 里用）
    dist = cv::Mat::zeros(1, DIST_COEFF_NUM, CV_64F);
}

PnpResult PnpSolver::solve(const std::vector<cv::Point2f>& img_ordered)
{

    PnpResult solve_result = {};  

    // 物点：装甲板的四个角，单位是米。
    // 顺序必须是 左上 → 右上 → 右下 → 左下，和图像里角点的顺序一一对应。
    // 这里要是错一位，解出来的位姿就会整个歪掉，而且不会报错，非常难查
    std::vector<cv::Point3f> obj_pts = {
        {-ARMOR_HALF_W, -ARMOR_HALF_H, ARMOR_PLANE_Z},   // 左上
        { ARMOR_HALF_W, -ARMOR_HALF_H, ARMOR_PLANE_Z},   // 右上
        { ARMOR_HALF_W,  ARMOR_HALF_H, ARMOR_PLANE_Z},   // 右下
        {-ARMOR_HALF_W,  ARMOR_HALF_H, ARMOR_PLANE_Z},   // 左下
    }; 

    //造两个空的 cv::Mat：rvec、tvec
    cv::Mat rvec;      
    cv::Mat tvec;

    //调用 cv::solvePnP
    // 用 SOLVEPNP_IPPE：这是专门给"平面目标"准备的方法，
    // 只有 4 个共面点的时候比通用方法稳定得多
    bool is_solve = cv::solvePnP(obj_pts, img_ordered, K, dist, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    if(!is_solve)
    {
        std::cout << "PnP解算失败！" << std::endl;
        return solve_result;
    }

    double z = tvec.at<double>(2, 0);

    // 角点顺序要是搞错了（比如连成了交叉的"蝴蝶形"），会解出一个在相机后面的解，
    // 表现出来就是 z 是负的。这里拦一下，免得后面算距离算出负值
    if(z <= 0)
    {
        std::cout << "PnP解算失败！（原因：深度为负，角点顺序可能有误）" << std::endl;
        return solve_result;
    }

    solve_result.ok = true;
    solve_result.rvec = rvec;
    solve_result.tvec = tvec;
    // distance 是相机到板心的直线距离，和 tvec 的 z（沿光轴的深度）不是一回事
    solve_result.distance = cv::norm(tvec);


    // 重投影误差：把解出来的位姿再投影回图像，和原来的角点比一比差多少像素。
    // 这个数可以用来发现"角点顺序错了"或者"内参不对"。
    // 但要注意它对平面目标的深度不敏感——平面上有很多个位姿都能投影出几乎一样的结果，
    // 所以判断距离准不准还是要靠真值，不能只看这个误差
    std::vector<cv::Point2f> predict_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K, dist, predict_pts);

    double sum = 0;
    for (size_t i = 0; i < img_ordered.size() && i < predict_pts.size(); ++i)
        sum += cv::norm(predict_pts[i] - img_ordered[i]);
    solve_result.reproj_error = predict_pts.empty() ? 0.0 : (sum / predict_pts.size());
    return solve_result;
}