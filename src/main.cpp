#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"
#include "pnp_solver.hpp"
#include "traditional_detector.hpp"
#include <cstdio>
#define ENABLE_DEBUG 0


namespace {

// 相机参数。
// 素材里有两套完全不同的相机：测试图片是 960×720 的仿真渲染（无畸变），
// 视频是 1440×1080 的真实拍摄（有畸变）。两套的 fx 差了 2.7 倍，
// 混用的后果是距离全部算错，而且程序不会报错，所以一定分开用。

// 测试图片：960×720，仿真渲染，无畸变
constexpr double IMG_FX = 869.1168824543144;
constexpr double IMG_FY = 869.1168824543144;
constexpr double IMG_CX = 480.0;
constexpr double IMG_CY = 360.0;

// 素材视频：1440×1080，真实相机
constexpr double VID_FX = 2362.2795952460974;
constexpr double VID_FY = 2336.1916035376717;
constexpr double VID_CX = 679.17513921636382;
constexpr double VID_CY = 529.06858905837635;

// 视频的畸变系数，顺序是 k1, k2, p1, p2, k3
constexpr double VID_K1 = -0.098561592108101559;
constexpr double VID_K2 =  0.11173066531801299;
constexpr double VID_P1 = -0.0038352945867108721;
constexpr double VID_P2 = -0.0020421443354436304;
constexpr double VID_K3 =  0.0;

// 输入输出路径。集中放在这里，换文件的时候只改这几行
constexpr const char* PATH_ARMOR_MODEL = "data/armor.onnx";
constexpr const char* PATH_DIGIT_MODEL = "data/mlp.onnx";
constexpr const char* PATH_VIDEO       = "data/video.avi";

constexpr const char* OUT_VIDEO_TRAD = "result_video_trad.avi";
constexpr const char* OUT_VIDEO_YOLO = "result_video_yolo.avi";

// 测试图片及与其一一对应的真值表
constexpr const char* TEST_IMAGES[] = { "data/test1.jpg",
                                        "data/test2.jpg",
                                        "data/test3.jpg" };
constexpr int TEST_IMAGE_COUNT = 3;   // 与 GT_OF_IMAGE 的元素个数必须一致

constexpr int    PROGRESS_EVERY_N = 30;      // 每处理 N 帧打印一次进度
constexpr double DIST_SENTINEL    = 1e9;     // 求最小值时的初始"无穷大"
constexpr int    TEXT_OFFSET_PX   = 18;      // 视频里距离文字相对框底的偏移
constexpr double FONT_SCALE       = 0.5;

} // namespace



// 打印视频的基本信息。主要是在开始处理前确认一下分辨率、帧数对不对，
// 如果这几项和预期不一样，说明读到的文件不对或者解码器有问题
static bool probeVideo(const std::string& path)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
    {
        std::printf("打不开视频: %s\n", path.c_str());
        return false;
    }

    double w      = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double h      = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps    = cap.get(cv::CAP_PROP_FPS);

    std::printf("视频: %s\n", path.c_str());
    std::printf("  分辨率: %.0f x %.0f\n", w, h);
    std::printf("  帧数  : %.0f\n", frames);
    std::printf("  帧率  : %.2f fps\n", fps);

    #if ENABLE_DEBUG
        //截取第0帧图片，调试用
        cv::Mat frame;
        if (cap.read(frame) && !frame.empty())
        {
            cv::imwrite("debug_video_frame0.png", frame);
            std::printf("  第 0 帧已存为 debug_video_frame0.png (%d x %d)\n",
                        frame.cols, frame.rows);
        }
    #endif
    cap.release();
    return true;
}




// 对一组检测结果做 PnP 解算并打印。
// 两种方案能共用这一个函数，是因为它们的输出类型一样（都是 vector<Detection>），
// 四个角点的顺序也一样（左上→右上→右下→左下）。
// 要是当初让两种方案各自输出自己的格式，这里就得写两份，还容易写不一致
static void solveAndPrint(const std::vector<Detection>& dets,
                          PnpSolver& solver,
                          const std::string& tag)
{
    for (const auto& d : dets)
    {
        // 解算：输入四个角点，输出位姿
        PnpResult r = solver.solve(d.kpt);

        std::cout << tag << "  cls=" << d.cls << "  score=" << d.score;

        // 解算失败就不用往下取 tvec 了（里面是空的）
        if (!r.ok)
        {
            std::cout << "  [解算失败]" << std::endl;
            continue;
        }

        std::cout << "  x=" << r.tvec.at<double>(0, 0)
                  << "  y=" << r.tvec.at<double>(1, 0)
                  << "  z=" << r.tvec.at<double>(2, 0)
                  << "  距离=" << r.distance << " m"
                  << "  误差=" << r.reproj_error << std::endl;
    }
}



// 真值表，数据来自素材里的 图1.md / 图2.md / 图3.md。
// 那几份文件里其实有 4 块板，但只有标了"正面"的两块在图里看得见，
// 所以这里只放正面那两块——把背面的也写进来，
// 我们的检测永远匹配不上，算出来的人为误差就全错了。
// 坐标是"相机光学系"下的，和 cv::solvePnP 输出的 tvec 是同一个坐标系，可以直接比
struct GtPlate //真值表
{
    const char* name;      // 板编号
    double      x, y, z;   // 真值位置（m）
};

static const std::vector<GtPlate> GT_OF_IMAGE[3] = {
    // test1.jpg → armor_1、armor_4 正面
    { {"armor_1", -0.174340, 0.169600, 3.794632},
      {"armor_4",  0.119301, 0.183937, 3.762585} },
    // test2.jpg → armor_1、armor_2 正面
    { {"armor_1",  0.285747, 0.070019, 1.384573},
      {"armor_2",  0.009916, 0.097664, 1.281592} },
    // test3.jpg → armor_3、armor_4 正面
    { {"armor_3",  0.189233, 0.070265, 1.553395},
      {"armor_4", -0.092414, 0.092079, 1.467008} },
};




// 给定一次解算结果，找出它对应哪块真值板：算三维距离，取最近的那个。
// 为什么不用图像位置来判断：一块板上可能同时有一个传统检测和一个神经网络检测，
// 而两块板在图上可能只隔一百多像素，靠眼睛或者比较图像坐标很容易配错。
// 只有三维坐标才能唯一确定"这次检测说的是哪块板"
static int findGtMatch(const cv::Mat& tvec, const std::vector<GtPlate>& gt)
{
    int    best   = -1;      // 目前最匹配的真值板下标
    double best_d = DIST_SENTINEL;     // 目前最小的三维距离（初值给一个很大的数）

    for (size_t i = 0; i < gt.size(); ++i)
    {
        // 计算 tvec 与这块真值板的三维差
        double dx = tvec.at<double>(0, 0) - gt[i].x;
        double dy = tvec.at<double>(1, 0) - gt[i].y;
        double dz = tvec.at<double>(2, 0) - gt[i].z;

        // 三维距离
        double d = std::sqrt(dx * dx + dy * dy + dz * dz);

        // 比当前最近还近，就替换
        if (d < best_d)
        {
            best_d = d;
            best   = static_cast<int>(i);
        }
    }
    return best;
}




// 打印一种方案在这张图上的误差表。
// 每一项都是"计算值 − 真值"，所以正数表示算偏了（远了/右了），负数表示算偏小了
static void printErrorTable(const std::vector<Detection>& dets,
                            PnpSolver& solver,
                            const std::vector<GtPlate>& gt,
                            const std::string& tag)
{

    for (const auto& d : dets)
    {
        //解算
        PnpResult r = solver.solve(d.kpt);
        if (!r.ok)
        {
            std::printf("  %s | [解算失败]\n", tag.c_str());
            continue;
        }

        //找到对应的真值板
        int gi = findGtMatch(r.tvec, gt);
        if (gi < 0) continue;

        //算误差（Δ = 计算值 − 真值）
        double dx = r.tvec.at<double>(0, 0) - gt[gi].x;
        double dy = r.tvec.at<double>(1, 0) - gt[gi].y;
        double dz = r.tvec.at<double>(2, 0) - gt[gi].z;

        // 三维误差：三个方向偏差的合成长度
        double e3 = std::sqrt(dx * dx + dy * dy + dz * dz);

        //打印一行
        std::printf("  %s | %-8s | dx=%+8.4f  dy=%+8.4f  dz=%+8.4f | %8.4f    | %+6.1f%%\n",
                    tag.c_str(),
                    gt[gi].name,
                    dx, dy, dz,
                    e3,
                    100.0 * dz / gt[gi].z);
    }
}


// 在视频上跑一种方案，输出一个标注好的视频。
// 处理顺序：读帧 → 畸变校正 → 检测 → 挑最近的 → 叠加距离文字 → 画框 → 写入。
// 注意是先校正再检测：这样检测和解算用的都是"无畸变"的图，
// 和解算器里畸变系数为零的写法对得上。反过来做（不校正、只在解算时传畸变系数）
// 就会出现"在畸变图上检测、用理想模型解算"的不一致。
static void runVideo(const std::string& in_path,
                     const std::string& out_path,
                     TraditionalDetector& trad,
                     YoloDetector& yolo,
                     PnpSolver& solver,
                     const cv::Mat& K, const cv::Mat& D,
                     bool use_trad)
{
    cv::VideoCapture cap(in_path);
    if (!cap.isOpened())
    {
        std::printf("打不开视频: %s\n", in_path.c_str());
        return;
    }

    const int w     = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h     = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int total = int(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::printf("输出视频: %s  (%d x %d, %.0f fps, 共 %d 帧)\n", out_path.c_str(), w, h, fps, total);

    // MJPG + .avi：兼容性最好，Windows 自带播放器也能直接放。
    // 用 mp4 的话要 OpenCV 编译时带了对应编码器，不一定有
    cv::VideoWriter writer(out_path,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(w, h));
    if (!writer.isOpened())
    {
        std::printf("创建视频失败: %s\n", out_path.c_str());
        return;
    }

    cv::Mat frame, frame_u;
    int idx = 0;

    while (cap.read(frame))
    {
        // 畸变校正，之后的处理全在 frame_u 上进行
        cv::undistort(frame, frame_u, K, D);

        // 检测。两种方案的 detect 签名一样，所以这里能用一个三元表达式带过
        std::vector<Detection> dets = use_trad ? trad.detect(frame_u)
                                               : yolo.detect(frame_u);


        // 只保留距离最近的那一块。
        // 真实比赛画面里同时有好几台车，检测层会把它们都检出来。
        // 而决策层（找最近的敌人打）只要一个目标，所以这里按距离挑最近的那块
        if (dets.size() > 1)
        {
            int    best_k = -1;
            double best_z = DIST_SENTINEL;

            for (size_t k = 0; k < dets.size(); ++k)
            {
                PnpResult r = solver.solve(dets[k].kpt);
                if (!r.ok) continue;
                double z = r.tvec.at<double>(2, 0);
                if (z < best_z) { best_z = z; best_k = int(k); }
            }
            if (best_k >= 0)
            {
                Detection keep = dets[best_k];
                dets.clear();
                dets.push_back(keep);
            }
        }

        // 把解算出来的距离也写到画面上，这样看视频的时候能直接看出远近变化
        for (size_t k = 0; k < dets.size(); k++)
        {
            PnpResult r = solver.solve(dets[k].kpt);
            if (!r.ok) continue;

            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.2fm", r.tvec.at<double>(2, 0));

            // 位置要和 draw() 里的标签错开，不然两个文字会叠在一起。
            // 传统方案的标签在框下方，所以距离文字放框上方；
            // 神经网络反过来，标签在上方，距离文字放框下方
            const int dist_y = use_trad ? int(dets[k].box.y) - 5
                                        : int(dets[k].box.y + dets[k].box.height) + TEXT_OFFSET_PX;

            // 颜色和框保持一致：传统黄色、神经网络绿色
            const cv::Scalar dist_color = use_trad ? cv::Scalar(0, 255, 255)
                                                  : cv::Scalar(0, 255, 0);

            cv::putText(frame_u, buf,
                        cv::Point(int(dets[k].box.x), dist_y),
                        cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, dist_color, 1);
        }

        // 画框
        if (use_trad) trad.draw(frame_u, dets);
        else          yolo.draw(frame_u, dets);

        writer.write(frame_u);

        ++idx;
        if (idx % PROGRESS_EVERY_N == 0)
            std::printf("  已处理 %d / %d 帧\n", idx, total);
    }

    writer.release();
    cap.release();
    std::printf("完成: %d 帧 → %s\n", idx, out_path.c_str());
}


int main()
{

    probeVideo(PATH_VIDEO);


    // 神经网络方案只有一个模型，和颜色、相机都没关系，建一个就够
    YoloDetector detector(PATH_ARMOR_MODEL);


    // 解算器要按相机分开建，因为两套内参差很多。
    // solver_img 给三张测试图片用，solver_vid 给视频用
    PnpSolver solver_img(IMG_FX, IMG_FY, IMG_CX, IMG_CY);
    PnpSolver solver_vid(VID_FX, VID_FY, VID_CX, VID_CY);

    // 传统方案要按装甲板颜色分开建。测试图片里是蓝色的板，视频里是红色的。
    // 两个网络权重是一样的，只是 binarize 里用哪个通道做差不同。
    // 颜色搞反的话 mask 会全黑，一根灯条都找不到，但程序不报错
    TraditionalDetector trad_img(true,  PATH_DIGIT_MODEL);
    TraditionalDetector trad_vid(false, PATH_DIGIT_MODEL);

    
    // 视频里有一小段时间把背景里人的胳膊误认成了装甲板，
    // 打开这个选项后，认不出数字的候选就直接丢掉（对应讲义里"判断中间是否有数字"这一步）。
    // 测试图片没开，因为远处的板太小、数字认不出来，开了反而会漏检
    trad_vid.setRequireDigit(true);

    // 编译期检查：测试图片数量必须与真值表数量一致。
    // 用 static_assert 而不是运行时 if，是因为数量对不上属于写代码时的错误，
    // 应该在编译阶段就发现，而不是运行时悄悄少打几行表
    static_assert(sizeof(TEST_IMAGES) / sizeof(TEST_IMAGES[0]) == TEST_IMAGE_COUNT,
                  "TEST_IMAGES 的元素个数必须等于 TEST_IMAGE_COUNT");
    static_assert(sizeof(GT_OF_IMAGE) / sizeof(GT_OF_IMAGE[0]) == TEST_IMAGE_COUNT,
                  "GT_OF_IMAGE 的元素个数必须等于 TEST_IMAGE_COUNT");

    // 处理测试图片并输出误差表
    for (size_t i = 0; i < TEST_IMAGE_COUNT; i++)
    {

        std::printf("\n================ 第 %d 张图：%s ================\n",
                     int(i) + 1, TEST_IMAGES[i]);

        cv::Mat img = cv::imread(TEST_IMAGES[i]);
        if (img.empty())
        {
            std::cout << "读不到图: " << TEST_IMAGES[i] << std::endl;
            continue;
        }

        // 传统视觉方案
        std::vector<Detection> trad_dets = trad_img.detect(img);
        solveAndPrint(trad_dets, solver_img, "[传统]");

        // 只画传统方案的黄框
        cv::Mat img_trad = img.clone();
        trad_img.draw(img_trad, trad_dets);

        // 神经网络方案
        std::vector<Detection> yolo_dets = detector.detect(img);
        solveAndPrint(yolo_dets, solver_img, "[神经]");

        // 只画神经网络的绿框
        cv::Mat img_yolo = img.clone();
        detector.draw(img_yolo, yolo_dets);

        // 叠加图：两种方案的框画在同一张上，用来看角点位置差多少。
        // 注意 cv::Mat 的赋值是浅拷贝（共用像素），
        // 所以三张图都必须 clone，不然画在一张上三张都会变
        cv::Mat img_cmp = img.clone();
        trad_img.draw(img_cmp, trad_dets);      // 先画粗的黄色
        detector.draw(img_cmp, yolo_dets);      // 再画细的绿色，压在黄线上面

        // 误差表：表头只打一次，然后两种方案各打自己的数据行
        std::printf("\n[误差表]\n");
        std::printf("  [方案] | [真值板]  |          Δx / Δy / Δz (m)          | 三维误差(m) |  Δz%%\n");
        printErrorTable(trad_dets, solver_img, GT_OF_IMAGE[i], "传统");
        printErrorTable(yolo_dets, solver_img, GT_OF_IMAGE[i], "神经");

        // 保存三个文件
        std::string img_file = "result_" + std::to_string(i) + ".jpg";
        std::string base = img_file.substr(0, img_file.find_last_of('.'));
        cv::imwrite(base + "_trad.jpg", img_trad);
        cv::imwrite(base + "_yolo.jpg", img_yolo);
        cv::imwrite(base + "_cmp.jpg",  img_cmp);
    }


    // 视频用的内参和畸变系数，写成 cv::Mat 是因为 cv::undistort 要这个格式
    cv::Mat K_video = (cv::Mat_<double>(3, 3) <<
                       VID_FX, 0.0,    VID_CX,
                       0.0,    VID_FY, VID_CY,
                       0.0,    0.0,    1.0);

    cv::Mat D_video = (cv::Mat_<double>(1, 5) << VID_K1, VID_K2, VID_P1, VID_P2, VID_K3);

    std::printf("\n================ 视频处理 ================\n");

    // 跑两个视频，一个方案一个。
    // 其实 531 帧要读两遍，有点浪费，但这样两个视频互不影响，代码也简单
    runVideo(PATH_VIDEO, OUT_VIDEO_TRAD, trad_vid, detector, solver_vid, K_video, D_video, true);
    runVideo(PATH_VIDEO, OUT_VIDEO_YOLO, trad_vid, detector, solver_vid, K_video, D_video, false);


    return 0;
}