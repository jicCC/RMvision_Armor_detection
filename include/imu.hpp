#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct ImuSample
{
    double t{0.0};
    double qw{1.0};
    double qx{0.0};
    double qy{0.0};
    double qz{0.0};
};

class ImuBuffer
{
    public:
        static constexpr double kReceiveOffset = 0.009;

        bool load(const std::string& path);
        size_t size() const { return s_.size(); }    
        double stampOf(size_t i) const;
        static cv::Mat quatToMat(double w, double x, double y, double z);
        static cv::Mat camToBody(int variant);
        cv::Mat rotationAt(double stamp, int mode) const;
        cv::Mat camToWorld(const cv::Mat& p_cam, double stamp, int mode, int variant) const;
        void verify(const std::vector<cv::Mat>& cam_pts, const std::vector<double>& stamps) const;

    private:
        std::vector<ImuSample> s_; 
};