#include "imu.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <algorithm>

bool ImuBuffer::load(const std::string& path)
{
    std::ifstream fin(path); //打开文件
    if (!fin)
    {
        std::printf("[imu] 打不开文件: %s\n", path.c_str());
        return false;
    }

    s_.clear(); //清空 vector,防止第二次调用 load 时旧数据还在
    std::string line;  

    while (std::getline(fin, line))
    {
        if (line.empty()) continue;

        std::istringstream ss(line); //把字符串 line 包装成输入流
        ImuSample s;
        if (!(ss >> s.t >> s.qw >> s.qx >> s.qy >> s.qz)) continue; //连续用 >> 从字符串里取 5 个数

        const double n = std::sqrt(s.qw*s.qw + s.qx*s.qx + s.qy*s.qy + s.qz*s.qz);
        if (n < 1e-9) continue;

        s.qw /= n; s.qx /= n; s.qy /= n; s.qz /= n;
        s_.push_back(s);
    }

    if (s_.size() >= 2)
    {
        const double span = s_.back().t - s_.front().t;
        std::printf("[imu] 载入 %zu 行，时间 %.3f ~ %.3f s（跨度 %.3f，平均间隔 %.4f ≈ %.1f Hz）\n",
                    s_.size(), s_.front().t, s_.back().t, span,
                    span / (s_.size() - 1), (s_.size() - 1) / span);
    }
    return !s_.empty();
}


double ImuBuffer::stampOf(size_t i) const //取第i行的时间戳
{
    if (s_.empty()) return 0.0;
    if (i >= s_.size()) i = s_.size() - 1;
    return s_[i].t; 
}


cv::Mat ImuBuffer::quatToMat(double w, double x, double y, double z) //四元数 → 旋转矩阵
{
    const double xx = x*x, yy = y*y, zz = z*z;
    const double xy = x*y, xz = x*z, yz = y*z;
    const double wx = w*x, wy = w*y, wz = w*z;

    return (cv::Mat_<double>(3, 3) <<
            1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),
            2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx),
            2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy));
}


cv::Mat ImuBuffer::camToBody(int variant)//相机系 → 机身系
{
    if (variant == 0)
        return (cv::Mat_<double>(3, 3) <<
                 0,  0,  1,
                -1,  0,  0,
                 0, -1,  0);

    return (cv::Mat_<double>(3, 3) <<
             0, 0, 1,
             1, 0, 0,
             0, 1, 0);
}

cv::Mat ImuBuffer::rotationAt(double stamp, int mode) const//查看某时刻的机身朝向
{
    if (s_.empty()) return cv::Mat::eye(3, 3, CV_64F);

    const double qt = stamp - kReceiveOffset;

    auto toMat = [mode](double w, double x, double y, double z) {
        const cv::Mat R = ImuBuffer::quatToMat(w, x, y, z);
        return (mode == 1) ? R.t() : R;
    };

    if (qt <= s_.front().t)
        return toMat(s_.front().qw, s_.front().qx, s_.front().qy, s_.front().qz);
    if (qt >= s_.back().t)
        return toMat(s_.back().qw, s_.back().qx, s_.back().qy, s_.back().qz);


    size_t k = 0;
    for (size_t i = 0; i + 1 < s_.size(); ++i)
    {
        if (s_[i].t <= qt && qt < s_[i + 1].t) { k = i; break; }
    }

    const ImuSample& a = s_[k];
    const ImuSample& b = s_[k + 1];

    const double den = b.t - a.t;
    const double u   = (den > 1e-9) ? (qt - a.t) / den : 0.0;

    double bw = b.qw, bx = b.qx, by = b.qy, bz = b.qz;
    const double dot = a.qw*bw + a.qx*bx + a.qy*by + a.qz*bz;
    if (dot < 0.0) { bw = -bw; bx = -bx; by = -by; bz = -bz; }

     double w = a.qw + u * (bw - a.qw);
    double x = a.qx + u * (bx - a.qx);
    double y = a.qy + u * (by - a.qy);
    double z = a.qz + u * (bz - a.qz);

    const double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n < 1e-9) return cv::Mat::eye(3, 3, CV_64F);
    w /= n; x /= n; y /= n; z /= n;

    return toMat(w, x, y, z);

}


cv::Mat ImuBuffer::camToWorld(const cv::Mat& p_cam, double stamp, int mode, int variant) const //从相机系到世界系
{
    const cv::Mat R_imu = rotationAt(stamp, mode);
    const cv::Mat R_cb  = camToBody(variant);
    return R_imu * R_cb * p_cam;
}


void ImuBuffer::verify(const std::vector<cv::Mat>& cam_pts,
                       const std::vector<double>&    stamps) const
{
    for (int mode = 0; mode < 2; ++mode)
    {
        for (int variant = 0; variant < 2; ++variant)
        {
            double zmin = 1e9, zmax = -1e9, rmin = 1e9, rmax = -1e9;
            for (size_t i = 0; i < cam_pts.size(); ++i)
            {
                const cv::Mat pw = camToWorld(cam_pts[i], stamps[i], mode, variant);
                const double z = pw.at<double>(2);
                const double r = std::sqrt(pw.at<double>(0)*pw.at<double>(0)
                                         + pw.at<double>(1)*pw.at<double>(1));
                zmin = std::min(zmin, z); zmax = std::max(zmax, z);
                rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            }
            const bool good = (std::abs(zmin) < 0.6 && std::abs(zmax) < 0.6 && rmax > 0.8);
            std::printf("  mode=%d  variant=%d : z=%.3f~%.3f  |xy|=%.3f~%.3f   %s\n",
                        mode, variant, zmin, zmax, rmin, rmax,
                        good ? "✓ 像" : "✗ 不像");
        }
    }
}