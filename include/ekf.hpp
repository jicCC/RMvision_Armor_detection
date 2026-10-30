#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>

//定义一组整数，包括机器人的性质
enum StateIdx { CX=0, VX, CY, VY, ZA, VZ, THETA, OMEGA, R_ARM, DIM=9 };

//定义4个装甲板相关几何常量
constexpr int PLATE_NUM = 4;   //装甲板数量为4
constexpr double PLATE_PHI[PLATE_NUM] = { 0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0 };   //四块板相对参考板的角度偏移
constexpr double PLATE_DH[PLATE_NUM]  = { 0, -0.05, 0, -0.05 };   //四块板相对参考板的高度差
constexpr double PLATE_RADIUS = 0.25;     //装甲板的半径

// 过程噪声：目标的机动能力
//   SIGMA_A 是"每帧未建模加速度"的标准差（m/s²），均值视为 0，
//   因为运动模型假设匀速，真实加速度是围绕 0 的随机偏差。
//   注意：由于各帧加速度相互独立，速度不确定度的增长是 √(T·dt) 关系，
//   会低估持续机动的漂移，因此实际取值需通过扫参确定（先试 3，必要时加大到 10~30）。
constexpr double SIGMA_A       = 1.0;    // 平移加速度标准差（m/s²）
constexpr double SIGMA_A_THETA = 1.0;    // 角加速度标准差（rad/s²）
constexpr double SIGMA_A_Z = 0.3;    // 竖直方向：车不会飞，加速度接近 0

class Ekf
{
public:
    Ekf();

    cv::Mat h(int j, const cv::Mat& x) const; //一块板的 (x, y, z)
    cv::Mat H(int j, const cv::Mat& x) const; //雅可比（27 个偏导）

    void init(double plate_x, double plate_y, double plate_z,
              double theta, double r, double pos_var);

    // 把朝向平移 dtheta（弧度），并适当放大朝向方差，供分支重收敛使用（此处借鉴sp_25vision开源）
    void shiftTheta(double dtheta);


    void predict(double dt);
    bool correct(int j, const cv::Mat& z, const cv::Mat& R);

    const cv::Mat& state() const { return x_; } 
    const cv::Mat& cov()   const { return P_; }

private:
    cv::Mat x_;  //滤波器自己维护的状态
    cv::Mat P_;  //协方差矩阵
};
