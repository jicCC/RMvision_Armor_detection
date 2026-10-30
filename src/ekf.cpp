#include "ekf.hpp"

Ekf::Ekf()
{
    x_ = cv::Mat::zeros(DIM,1,CV_64F); //x_ 变成 9 行 1 列、全 0 的矩阵

    x_.at<double>(ZA) = 0.25;
    x_.at<double>(R_ARM) = PLATE_RADIUS;

    P_ = cv::Mat::eye(DIM,DIM,CV_64F) * 10.0; //P_ 变成 9 行 9 列、对角线上是 10、其余是 0 的矩阵

    P_.at<double>(R_ARM,R_ARM) = 1e-4;
}


// 用一次"某块板的位置观测"来初始化状态
// theta 是猜测的朝向，pos_var 是初始位置方差
void Ekf::init(double plate_x, double plate_y, double plate_z,
               double theta, double r, double pos_var)
{
    x_ = cv::Mat::zeros(DIM, 1, CV_64F);

    const double th = theta + PLATE_PHI[0];   // 这里假设观测的是板 0

    x_.at<double>(CX)    = plate_x + r * std::cos(th);
    x_.at<double>(CY)    = plate_y + r * std::sin(th);
    x_.at<double>(ZA)    = plate_z - PLATE_DH[0];
    x_.at<double>(THETA) = theta;
    x_.at<double>(R_ARM) = r;

    P_ = cv::Mat::eye(DIM, DIM, CV_64F) * pos_var;
    P_.at<double>(THETA, THETA) = 1.0;        // 朝向比位置更不确定
    P_.at<double>(OMEGA, OMEGA) = 4.0;
    P_.at<double>(R_ARM, R_ARM) = 1e-8;       // 半径几乎是已知的
}


cv::Mat Ekf::h(int j, const cv::Mat& x) const
{
    const double th = x.at<double>(THETA) + PLATE_PHI[j];

    const double r = x.at<double>(R_ARM);

    //返回板的x,y,z
    return (cv::Mat_<double>(3,1) << 
            x.at<double>(CX) - r * std::cos(th),
            x.at<double>(CY) - r * std::sin(th),
            x.at<double>(ZA) + PLATE_DH[j]      );
}


cv::Mat Ekf::H(int j, const cv::Mat& x) const
{
    const double th = x.at<double>(THETA) + PLATE_PHI[j];

    const double r = x.at<double>(R_ARM);
    const double s = std::sin(th);
    const double c = std::cos(th);

    cv::Mat Hm = cv::Mat::zeros(3, DIM, CV_64F);

    Hm.at<double>(0, CX) = 1;
    Hm.at<double>(0, THETA) = r*s;
    Hm.at<double>(0, R_ARM) = -c;
    Hm.at<double>(1, CY) = 1;
    Hm.at<double>(1, THETA) = -r*c;
    Hm.at<double>(1, R_ARM) = -s;
    Hm.at<double>(2, ZA) = 1;

    return Hm;
}

void Ekf::predict(double dt)
{
    cv::Mat F = cv::Mat::eye(DIM, DIM, CV_64F);  //F:这一帧推导到下一帧的换算表

    F.at<double>(CX,    VX)    = dt;
    F.at<double>(CY,    VY)    = dt;
    F.at<double>(ZA,    VZ)    = dt;
    F.at<double>(THETA, OMEGA) = dt;

    cv::Mat G = cv::Mat::zeros(DIM, 4, CV_64F);  //G:加速度 → 状态 的换算表

    const double h = 0.5 * dt * dt;      //x = dt²/2 * a
    const double v = dt;                 //v = a * dt

    // 行 CX
    G.at<double>(CX, 0) = h;
    // 行 VX
    G.at<double>(VX, 0) = v;
    // 行 CY
    G.at<double>(CY, 1) = h;
    // 行 VY
    G.at<double>(VY, 1) = v;
    // 行 ZA
    G.at<double>(ZA, 2) = h;
    // 行 VZ
    G.at<double>(VZ, 2) = v;
    // 行 THETA
    G.at<double>(THETA, 3) = h;
    // 行 OMEGA
    G.at<double>(OMEGA, 3) = v;

    cv::Mat sig = (cv::Mat_<double>(4, 1) <<
               SIGMA_A * SIGMA_A,
               SIGMA_A * SIGMA_A,
               SIGMA_A_Z * SIGMA_A_Z,    // z 通道用更小的 σa_z
               SIGMA_A_THETA * SIGMA_A_THETA);

    cv::Mat Sig = cv::Mat::diag(sig); //sig:加速度有多大

    cv::Mat Q = G * Sig * G.t(); //Q:过程噪声协方差

    Q.at<double>(R_ARM, R_ARM) = 0.0; //给r加一个极小的噪声，使其可以微调修正（修改：不加任何噪声）

    //更新x_和P_
    x_ = F * x_;
    P_ = F * P_ * F.t() + Q;

    while (x_.at<double>(THETA) >  M_PI) x_.at<double>(THETA) -= 2.0 * M_PI;
    while (x_.at<double>(THETA) < -M_PI) x_.at<double>(THETA) += 2.0 * M_PI;


    

    P_ = 0.5 * (P_ + P_.t());
}

bool Ekf::correct(int j, const cv::Mat& z, const cv::Mat& R)
{
    cv::Mat hx = h(j, x_);  //hx:观测的预测

    cv::Mat e = z - hx;  //e:残差

    cv::Mat Hm = H(j, x_); //Hm:雅可比

    cv::Mat S = Hm * P_ * Hm.t() + R; //S:新息协方差

    cv::Mat Sinv;
    cv::invert(S, Sinv, cv::DECOMP_CHOLESKY);      // 求 S 的逆

    cv::Mat K = P_ * Hm.t() * Sinv;  //K:卡尔曼增益

        // 马氏距离平方（NIS）
    cv::Mat nis = e.t() * Sinv * e;          // (1×3)(3×3)(3×1) = 1×1 ✓
    double d2 = nis.at<double>(0);

    if (d2 > 11.34)      // 3 自由度卡方 99% 分位
        return false;    // 拒绝这次观测

    x_ = x_ + K * e; //更新x_

    //Joseph形式更新协方差
    cv::Mat IKH = cv::Mat::eye(DIM, DIM, CV_64F) - K * Hm;
    P_ = IKH * P_ * IKH.t() + K * R * K.t();

    P_ = 0.5 * (P_ + P_.t());

    return true;
}


// 把朝向平移 dtheta，并把朝向方差抬到至少 0.5，让后续观测能快速纠正
void Ekf::shiftTheta(double dtheta)
{
    //平移
    x_.at<double>(THETA) += dtheta;

    //(-pi, pi]
    while (x_.at<double>(THETA) >  M_PI) x_.at<double>(THETA) -= 2.0 * M_PI;
    while (x_.at<double>(THETA) < -M_PI) x_.at<double>(THETA) += 2.0 * M_PI;

    //放大朝向方差
    const double cur = P_.at<double>(THETA, THETA);
    if (cur < 0.5) P_.at<double>(THETA, THETA) = 0.5;
}