#include "yolo_detector.hpp"        
#include "pnp_solver.hpp"           
#include "imu.hpp"                 
#include "armor_tracker.hpp"        
#include "ekf.hpp"                  
#include <opencv2/opencv.hpp>      
#include <cstdio>                   
#include <vector>                   
#include <string>        
#include <cmath>     


constexpr int LINE_THICK = 2;
constexpr int BOX_THICK = 2;      // 重投影框的线宽

constexpr double VID_FX = 2362.2795952460974;
constexpr double VID_FY = 2336.1916035376717;
constexpr double VID_CX = 679.17513921636382;
constexpr double VID_CY = 529.06858905837635;

constexpr double ARMOR_HALF_W = 0.0675;    
constexpr double ARMOR_HALF_H = 0.0275;    

//在视频右上角显示当前状态量
static void drawHud(cv::Mat& img, const Ekf& ekf, int j)
{
    const int W = img.cols;
    const int boxW = 300;
    const int boxH = 150;
    const cv::Point tl(W - boxW - 12, 12);
    const cv::Point br(W - 12, 12 + boxH);

    cv::rectangle(img, tl, br, cv::Scalar(40, 40, 40), -1);

    

    const cv::Mat s = ekf.state(); 
    const double cx = s.at<double>(CX);
    const double cy = s.at<double>(CY);
    const double vx = s.at<double>(VX);
    const double vy = s.at<double>(VY);
    const double za = s.at<double>(ZA);
    const double vz = s.at<double>(VZ);
    const double th = s.at<double>(THETA);
    const double om = s.at<double>(OMEGA);
    const double r = s.at<double>(R_ARM);

    char buf[80];


    int y = tl.y + 26;      // 第一行文字的基线位置


    snprintf(buf, sizeof(buf), "cx %+7.3f cy %+7.3f", cx, cy);
    cv::putText(img, buf, cv::Point(tl.x + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    y += 22;

    snprintf(buf, sizeof(buf), "vx %+7.3f vy %+7.3f", vx, vy);
    cv::putText(img, buf, cv::Point(tl.x + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    y += 22;

    snprintf(buf, sizeof(buf), "za %+7.3f vz %+7.3f", za, vz);
    cv::putText(img, buf, cv::Point(tl.x + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    y += 22;

    snprintf(buf, sizeof(buf), "th %+7.3f om %+7.3f", th, om);
    cv::putText(img, buf, cv::Point(tl.x + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    y += 22;

    snprintf(buf, sizeof(buf), "r %7.3f j %2d", r, j);
    cv::putText(img, buf, cv::Point(tl.x + 10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1, cv::LINE_AA);
    y += 22;

}





//算四个角点
static void plateCorners(const Ekf& ekf,
                         int j,
                         double thetaUse,
                         std::vector<cv::Point3d>& out)
{
    out.clear();

    const double thj = thetaUse + PLATE_PHI[j];

    const cv::Mat st = ekf.state();


    double cj = std::cos(thj);
    double sj = std::sin(thj);


    //算板中心
    const cv::Mat P = ekf.h(j, st);
    const double Px = P.at<double>(0);
    const double Py = P.at<double>(1);
    const double Pz = P.at<double>(2);

    //三个偏移量
    const double dx = ARMOR_HALF_W * sj;    
    const double dy = ARMOR_HALF_W * cj;
    const double dz = ARMOR_HALF_H;


    //算四个角点
    out.push_back(cv::Point3d(Px - dx, Py + dy, Pz + dz));  
    out.push_back(cv::Point3d(Px + dx, Py - dy, Pz + dz));  
    out.push_back(cv::Point3d(Px + dx, Py - dy, Pz - dz));  
    out.push_back(cv::Point3d(Px - dx, Py + dy, Pz - dz));  


}


//画框
static void drawArmorBoxes(cv::Mat& img,
                           const Ekf& ekf,
                           const ImuBuffer& imu,
                           double t,
                           int jMatched,
                           const cv::Mat& rvecMeas)
{   
    //转化参考系：世界系到相机系的旋转矩阵
    cv::Mat R_cw = (imu.rotationAt(t, 0) * ImuBuffer::camToBody(0)).t();

    // 默认用 EKF 估计的 θ
    double thUse = ekf.state().at<double>(THETA);

    // 如果这一帧有 PnP 结果，就用板的真实法线反解 θ
    if (!rvecMeas.empty() && jMatched >= 0)
    {
        cv::Mat R_pc;
        cv::Rodrigues(rvecMeas, R_pc);                          // 物体系 -> 相机系
        const cv::Mat nCam = R_pc.col(2);                       // 板法线（相机系）
        const cv::Mat R_wc = imu.rotationAt(t, 0) * ImuBuffer::camToBody(0);  // 相机 -> 世界（不转置）
        const cv::Mat nW   = R_wc * nCam;                       // 板法线（世界系）

        const double nx = nW.at<double>(0);
        const double ny = nW.at<double>(1);

        if (std::sqrt(nx * nx + ny * ny) > 1e-6)                // 防止法线接近竖直
            thUse = std::atan2(-ny, -nx) - PLATE_PHI[jMatched];
    }

    std::vector<cv::Point3d> corners;

    for(int j = 0; j < PLATE_NUM; ++j)
    {

        plateCorners(ekf, j, thUse, corners);

        std::vector<cv::Point> pts;
        bool ok = true;

        for (const cv::Point3d& c3d : corners)
        {
            const cv::Mat p3 = (cv::Mat_<double>(3, 1) << c3d.x, c3d.y, c3d.z);

            const cv::Mat pc = R_cw * p3; //世界系转化到相机系

            //取出三个分量
            const double Xc = pc.at<double>(0);
            const double Yc = pc.at<double>(1);
            const double Zc = pc.at<double>(2);

            if (Zc <= 0.1) { ok = false; break; } //处理非正深度异常

            //重投影
            const int px = cvRound(VID_FX * Xc / Zc + VID_CX);
            const int py = cvRound(VID_FY * Yc / Zc + VID_CY);

            //图像边界检查：有一个点跳出屏幕外了就不画框
            if (px < 0 || px >= img.cols || py < 0 || py >= img.rows)
            { 
                ok = false; 
                break; 
            }

            pts.push_back(cv::Point(px, py));
        }

        if (!ok || pts.size() != 4) //写pts.size() != 4的原因：防御性编程，防止被改内层循环逻辑
            continue;

        //画框
        const cv::Scalar color = (j == jMatched) ? cv::Scalar(0, 255, 0)
                                                 : cv::Scalar(90, 90, 90);
        const int thick = (j == jMatched) ? (BOX_THICK + 1) : BOX_THICK;

        cv::polylines(img, pts, true, color, thick, cv::LINE_AA);
    }
}

int main()
{

    //1.载入IMU
    ImuBuffer imu;
    if (!imu.load("data/imu.txt")) return 1;

    //2.载入神经+PnP
    YoloDetector yolo("data/armor.onnx");
    PnpSolver solver(VID_FX, VID_FY, VID_CX, VID_CY);

    //3.打开视频并提供输出窗口
    cv::VideoCapture cap("data/video.avi");
    if (!cap.isOpened())
    {
        printf("打不开视频: data/video.avi\n");
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS); //帧率
    const int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);//宽
    const int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);//高

    printf("[video] %.0f fps, %dx%d\n", fps, w, h);

    cv::VideoWriter writer("result_video_ekf.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(w, h)); //创建writer

    //4.载入Tracker
    ArmorTracker tracker(11.34, 0.10);

    cv::Mat frame;
    int idx = 0; //帧号
    int n_matched = 0; //匹配成功计数
    int n_miss = 0; //未匹配计数
    int last_j = -1; //上一帧板号
    int switches = 0; //识别的装甲板切换次数
    int n_pnp_fail = 0;      // 有检测但 PnP 失败的帧数
    
    const cv::Point2f ctr(w / 2.0f, h / 2.0f); //图像中心位置

    //打开meas.csv
    FILE* fcsv = std::fopen("meas.csv", "w");
    if (fcsv) std::fprintf(fcsv, "frame,x,y,z,ok\n");

    // 状态导出（供 plot_states.py 使用，列格式与 ekf_test 一致）
    FILE* fstate = std::fopen("state.csv", "w");
    if (fstate)
        std::fprintf(fstate,
            "frame,t,cx,vx,cy,vy,za,vz,theta,omega,r,matched,P_cx,P_cy,P_theta,raw_z\n");

    //5.对每一帧识别解算并预测画框
    while (cap.read(frame))
    {  
        //5.1神经网络识别
        std::vector<Detection> dets = yolo.detect(frame);
        
        //5.2挑选离图像中心最近的装甲板
        int best = -1;
        double best_d2 = 1e18;

        for (int i = 0; i < static_cast<int>(dets.size()); ++i)
        {
            if (dets[i].kpt.empty()) continue; //防越界



            //算识别框四个点的中心
            cv::Point2f c(0.0f, 0.0f);                  
            for (const cv::Point2f& p : dets[i].kpt)
            {
                c.x += p.x;
                c.y += p.y;
            }
            c.x /= static_cast<float>(dets[i].kpt.size());
            c.y /= static_cast<float>(dets[i].kpt.size());

            //算距离平方
            const double d2 = (c.x - ctr.x) * (c.x - ctr.x)
                        + (c.y - ctr.y) * (c.y - ctr.y);

            //更新最优
            if (d2 < best_d2) { best_d2 = d2; best = i; }
        }

        //5.3 PnP解算
        PnpResult pr;
        if (best >= 0)
            pr = solver.solve(dets[best].kpt);

        if (best >= 0 && !pr.ok) ++n_pnp_fail;    
        

        //5.4 转为世界系+喂给tracker
        const double t = imu.stampOf(idx); //时间戳
        std::vector<PlateMeas> ms; //观测容器
        // 由 rvec 反解板的方位角 θ_j（用于初始化）
        double thjMeas = 0.0;
        bool   hasThj  = false;

        if (pr.ok)
        {
            cv::Mat R_pc;
            cv::Rodrigues(pr.rvec, R_pc);
            const cv::Mat nCam = R_pc.col(2);
            const cv::Mat R_wc = imu.rotationAt(t, 0) * ImuBuffer::camToBody(0);
            const cv::Mat nW   = R_wc * nCam;

            const double nx = nW.at<double>(0);
            const double ny = nW.at<double>(1);

            if (std::sqrt(nx * nx + ny * ny) > 1e-6)
            {
                thjMeas = std::atan2(-ny, -nx);
                hasThj  = true;
            }
        }


        if (pr.ok)
        {
            PlateMeas m;
            m.thetaMeas = thjMeas;
            m.hasTheta  = hasThj;
            m.pos   = imu.camToWorld(pr.tvec, t, 0, 0);
            m.digit = (best >= 0) ? dets[best].cls   : -1;
            m.score = (best >= 0) ? dets[best].score : 0.0f;
            ms.push_back(m);
        }
        const int j = tracker.update(ms, t); //喂给Tracker

        if (fstate)
        {
            const cv::Mat sx = tracker.ekf().state();
            const cv::Mat Px = tracker.ekf().cov();
            std::fprintf(fstate,
                "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%d,%.6e,%.6e,%.6e,%.6f\n",
                idx, t,
                sx.at<double>(CX), sx.at<double>(VX),
                sx.at<double>(CY), sx.at<double>(VY),
                sx.at<double>(ZA), sx.at<double>(VZ),
                sx.at<double>(THETA), sx.at<double>(OMEGA),
                sx.at<double>(R_ARM),
                j,
                Px.at<double>(CX, CX),
                Px.at<double>(CY, CY),
                Px.at<double>(THETA, THETA),
                pr.ok ? pr.tvec.at<double>(2) : 0.0);
        }


        if (fcsv)
        {
            if (pr.ok)
                std::fprintf(fcsv, "%d,%.6f,%.6f,%.6f,1\n", idx,
                             pr.tvec.at<double>(0),
                             pr.tvec.at<double>(1),
                             pr.tvec.at<double>(2));
            else
                std::fprintf(fcsv, "%d,0,0,0,0\n", idx);
        }

        //5.5统计匹配情况
        if (j >= 0) 
        {
             ++n_matched; 

             if (last_j >= 0 && j != last_j) 
                ++switches; 

            last_j = j; 
        }
        else
        {
            ++n_miss;
        }


        //5.6 打印
        if (idx % 50 == 0)
        {
            const cv::Mat s = tracker.ekf().state();
            const double cx = s.at<double>(CX);
            const double cy = s.at<double>(CY);
            const double th = s.at<double>(THETA);

            printf(" 帧%4d t=%5.2f cx=%7.3f cy=%7.3f th=%6.3f j=%d\n", idx, t, cx, cy, th, j);
        }

        drawHud(frame, tracker.ekf(), j);

        if (tracker.inited())
        {
            drawArmorBoxes(frame, tracker.ekf(), imu, t, j, pr.ok ? pr.rvec : cv::Mat());
        }

        writer.write(frame);
        ++idx;
    }


    if (fcsv) std::fclose(fcsv);

    if (fstate) std::fclose(fstate);

    
    cap.release();
    writer.release();

    //6.最终统计打印
    printf("\n=== 统计 ===\n");
    printf("  总帧数    : %d\n", idx);
    printf("  匹配成功  : %d  (%.1f%%)\n", n_matched,
           100.0 * n_matched / (idx > 0 ? idx : 1));
    printf("  未匹配    : %d  (其中 PnP 失败 %d)\n", n_miss, n_pnp_fail);
    printf("  板号切换  : %d\n", switches);
    printf("  → result_video_ekf.avi 已输出\n");


    return 0; 
}