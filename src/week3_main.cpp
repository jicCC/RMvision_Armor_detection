#include "yolo_detector.hpp"        
#include "pnp_solver.hpp"           
#include "imu.hpp"                 
#include "armor_tracker.hpp"        
#include "ekf.hpp"                  
#include <opencv2/opencv.hpp>      
#include <cstdio>                   
#include <vector>                   
#include <string>            



constexpr double VID_FX = 2362.2795952460974;
constexpr double VID_FY = 2336.1916035376717;
constexpr double VID_CX = 679.17513921636382;
constexpr double VID_CY = 529.06858905837635;

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




static void drawArmorCenters(cv::Mat& img, const Ekf& ekf,
                             const ImuBuffer& imu, double t, int jMatched)
{
    cv::Mat R_cw = (imu.rotationAt(t, 0) * ImuBuffer::camToBody(0)).t();
    const cv::Mat s = ekf.state();
    for (int j = 0; j < PLATE_NUM; ++j)
    {
        cv::Mat pw = ekf.h(j, s);
        cv::Mat pc = R_cw * pw;
        const double Xc = pc.at<double>(0);
        const double Yc = pc.at<double>(1);
        const double Zc = pc.at<double>(2);


        if (Zc <= 0.1) continue;

        const int u = cvRound(VID_FX*Xc/Zc + VID_CX);
        const int v = cvRound(VID_FY*Yc/Zc + VID_CY);

        const cv::Scalar color = (j == jMatched)
                                 ? cv::Scalar(0, 255, 0)     // 当前匹配：绿
                                 : cv::Scalar(0, 165, 255);  // 其余：橙

        if (u < 0 || u >= img.cols || v < 0 || v >= img.rows) continue; //投影点必须在图像内
        cv::circle(img, cv::Point(u,v), 4, color, -1);
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
        if (pr.ok)
        {
            PlateMeas m;
            m.pos   = imu.camToWorld(pr.tvec, t, 0, 0);
            m.digit = (best >= 0) ? dets[best].cls   : -1;
            m.score = (best >= 0) ? dets[best].score : 0.0f;
            ms.push_back(m);
        }
        const int j = tracker.update(ms, t); //喂给Tracker

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


        //临时打印
        if (idx < 3 && pr.ok)
        {
        const cv::Mat pw = imu.camToWorld(pr.tvec, t, 0, 0);
        printf("  帧%d  tvec=(%.3f,%.3f,%.3f)  world=(%.3f,%.3f,%.3f)\n",
               idx,
               pr.tvec.at<double>(0), pr.tvec.at<double>(1), pr.tvec.at<double>(2),
               pw.at<double>(0),     pw.at<double>(1),     pw.at<double>(2));
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
             drawArmorCenters(frame, tracker.ekf(), imu, t, j);

        writer.write(frame);
        ++idx;
    }


    if (fcsv) std::fclose(fcsv);

    
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