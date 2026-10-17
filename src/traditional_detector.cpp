#include "traditional_detector.hpp"
#include <iostream>     
#include <algorithm>    
#include <cmath>     
#include <memory>   
#include <openvino/openvino.hpp>  
 

TraditionalDetector::TraditionalDetector(bool is_blue, const std::string& digit_model_path)
    : is_blue_(is_blue)
{
    // 加载数字识别网络
    std::cout << "正在加载数字网络: " << digit_model_path << std::endl;

    std::shared_ptr<ov::Model> model = digit_core_.read_model(digit_model_path);
    digit_model_   = digit_core_.compile_model(model, "CPU");
    digit_request_ = digit_model_.create_infer_request();
}

cv::Mat TraditionalDetector::binarize(const cv::Mat& img)
{
    //分离R G B三通道
    std::vector<cv::Mat> ch;
    cv::split(img, ch);   // ch[0]=B, ch[1]=G, ch[2]=R


    //算出"目标通道-另外两个通道的最大值"
    cv::Mat weak, score;
    if (is_blue_) 
    {
        cv::max(ch[2], ch[1], weak);        // weak = max(R, G)
        cv::subtract(ch[0], weak, score);   // score = B - max(R,G)

    } 
    else 
    {
        cv::max(ch[0], ch[1], weak);        // weak = max(B, G)
        cv::subtract(ch[2], weak, score);   // score = R - max(B,G)
    }

    cv::imwrite("debug_score.jpg", score);      // 调试用：看每根灯条到底有多"蓝"
    //两个条件同时满足才算灯条像素
    cv::Mat m1, m2, mask;
    cv::threshold(score, m1, 20, 255, cv::THRESH_BINARY);                  // 通道差 > 20
    cv::threshold(ch[is_blue_ ? 0 : 2], m2, 90, 255, cv::THRESH_BINARY);   // 亮度 > 90
    cv::bitwise_and(m1, m2, mask);

    //形态学闭运算：把灯条亮芯的断裂连成一整块
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);


    return mask;
}



std::vector<LightBar> TraditionalDetector::findLightBars(const cv::Mat& mask)
{
    std::vector<std::vector<cv::Point>> contours;   // 轮廓容器
    std::vector<cv::Vec4i>              hierarchy;  // 层级容器
    std::vector<LightBar>               bars; //灯条容器
    


    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    

    for (size_t i = 0; i < contours.size(); i++) // 用 contours[i] 取当前轮廓
    {
        LightBar bar;

        //用面积筛选灯条
        double area = cv::contourArea(contours[i]);

        if(area < 12)
            continue;
        //画出最小矩形
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        //取矩形的四个端点
        cv::Point2f rect_pts[4];
        rect.points(rect_pts);

        //算出相邻两条边的长度
        float e1 = cv::norm(rect_pts[1] - rect_pts[0]);
        float e2 = cv::norm(rect_pts[2] - rect_pts[1]);

        //比较这两条边哪条长，哪条短，长的存入length，短的存入width
        float length = std::max(e1,e2);
        float width  = std::min(e1,e2);

        //得到"长边向量"
        cv::Point2f edge; 

        if(e1>e2)
            edge = rect_pts[1] - rect_pts[0]; 
        else
            edge = rect_pts[2] - rect_pts[1]; 
        
        //算出夹角
        float tilt = std::atan2(std::abs(edge.x), std::abs(edge.y)) * 180.0 / CV_PI;

        //算出填充率（同时排除除以0的情况）
        if(length * width == 0) continue;

        double fill_rate = area / (length * width);

        /*筛选灯条(共5个条件)
        长宽比 > 1.5,长边像素长	≥ 6,填充率> 0.4,长轴与竖直方向夹角< 35°,短边（宽度）≥ 2*/
        /*std::cout << "候选: area=" << area
                  << " 长=" << length << " 宽=" << width
                  << " 长宽比=" << length / width
                  << " 填充=" << fill_rate
                  << " 倾角=" << tilt;
        if (length / width <= 1.5) std::cout << "  ✗长宽比不够";
        if (length < 6)            std::cout << "  ✗太短";
        if (fill_rate <= 0.5)      std::cout << "  ✗不够实心";
        if (tilt >= 35)            std::cout << "  ✗太横";
        if (width < 2)             std::cout << "  ✗太窄";
        std::cout << std::endl;*/
        if(length/width <= 1.5 || length < 6 || fill_rate <= 0.4 || tilt >= 35 || width < 2) continue;

        //定义两个角点，上角点和下角点
        cv::Point2f bar_pts1, bar_pts2; 
        cv::Point2f top_pts,bottom_pts;

        //算出两个角点
        if(e1>e2)
        {
            bar_pts1 = cv::Point2f((rect_pts[2].x +rect_pts[1].x) / 2.0, (rect_pts[2].y +rect_pts[1].y) / 2.0);
            bar_pts2 = cv::Point2f((rect_pts[3].x +rect_pts[0].x) / 2.0, (rect_pts[3].y +rect_pts[0].y) / 2.0); 
        }
        else
        {
            bar_pts1 = cv::Point2f((rect_pts[1].x +rect_pts[0].x) / 2.0, (rect_pts[1].y +rect_pts[0].y) / 2.0);
            bar_pts2 = cv::Point2f((rect_pts[3].x +rect_pts[2].x) / 2.0, (rect_pts[3].y +rect_pts[2].y) / 2.0);
        }
        
        //比较两个角点的纵坐标，区分出上下角点
        if(bar_pts1.y < bar_pts2.y)
        {
            top_pts = bar_pts1;
            bottom_pts = bar_pts2;
        }
        else
        {
            top_pts = bar_pts2;
            bottom_pts = bar_pts1;
        }

        //填 LightBar
        bar.center = rect.center;
        bar.length = length;
        bar.width = width;
        bar.angle = tilt;
        bar.top = top_pts;
        bar.bottom = bottom_pts;

        bars.push_back(bar);
    }


    return bars;
}   



static float pairCost(const LightBar& a, const LightBar& b) //给一对灯条打分
{
    //退化的灯条直接判不合格
    if (a.length <= 0.0f || b.length <= 0.0f)
        return 1000.0f;

    /*
    下面三个变量是为灯条打分所需的变量之三
    */

    //平均灯条长
    float avg_len = (a.length + b.length) / 2.0f;

    //灯条中心距
    float dist = cv::norm(a.center - b.center);

    // 距离比 = 中心距 ÷ 平均灯条长度
    float len_ratio = dist / avg_len;

    /*
    下面是判据
    */

    //判据1：两根灯条长度比不能过大
    float longer  = std::max(a.length, b.length);
    float shorter = std::min(a.length, b.length);
    if (longer / shorter >= 1.5f)
        return 1000.0f;

    //判据2：距离比要落在合理区间
    if (len_ratio < 1.5f || len_ratio > 3.6f)
        return 1000.0f;

    //判据3：两根灯条要近似平行
    if (std::abs(a.angle - b.angle) >= 12.0f)
        return 1000.0f;

    //判据4：两根灯条的高度要接近（两根灯条中心的纵坐标差应该远小于灯条长度）
    if (std::abs(a.center.y - b.center.y) / avg_len >= 0.5f)
        return 1000.0f;

    //上述4个判据全部通过后，返回成绩（代价）：数字越小越像一个装甲板
    return std::abs(len_ratio - 2.45f) / 2.45f;
}



int TraditionalDetector::recognizeDigit(const cv::Mat& img,
                                       const LightBar& left,
                                       const LightBar& right)
{
    // ==================== ① 把装甲板转正，再截取 ROI ====================
    cv::Point2f center((left.center.x + right.center.x) / 2.0f,
                       (left.center.y + right.center.y) / 2.0f);

    float bar_dist = cv::norm(left.center - right.center);
    float avg_len  = (left.length + right.length) / 2.0f;

    // 两灯条中心连线的倾角 = 装甲板的倾斜角
    float deg = std::atan2(right.center.y - left.center.y,
                           right.center.x - left.center.x) * 180.0 / CV_PI;

    // 绕 center 旋转把板摆平（绕 center 转 → center 位置不变，可继续用它截取）
    cv::Mat rot = cv::getRotationMatrix2D(center, deg, 1.0);
    cv::Mat upright;
    cv::warpAffine(img, upright, rot, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // 高度 = 灯条长 × 1.7（数字比灯条高，必须留余量，否则上下被切）
    const float ROI_H_SCALE = 1.9f;
    int roi_h = int(avg_len * ROI_H_SCALE);

    // 宽度按目标比例推（28:20 = 1.4），再限制不超过灯条间距的 0.6 倍（避免框进灯条）
    int roi_w = int(roi_h * 28.0f / 20.0f);
    if (roi_w > int(bar_dist * 0.6f)) roi_w = int(bar_dist * 0.6f);

    if (roi_w < 4 || roi_h < 4) return 0;

    cv::Mat roi;
    cv::getRectSubPix(upright, cv::Size(roi_w, roi_h), center, roi);
    if (roi.empty()) return 0;

    // 灰度 → 缩放 → Otsu 二值化 → 归一化
    cv::Mat gray, resized, bin, blob;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(20, 28));
    cv::threshold(resized, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  // 关键
    bin.convertTo(blob, CV_32F, 1.0 / 255.0);

    // 推理
    ov::Tensor input(ov::element::f32, {1, 20, 28, 1}, blob.ptr<float>());
    digit_request_.set_input_tensor(input);
    digit_request_.infer();
    const float* out = digit_request_.get_output_tensor().data<const float>();

    int best = 0;
    for (int k = 1; k < 9; ++k) if (out[k] > out[best]) best = k;

    // 调试：把二值化后的输入存下来（放大 10 倍，不然 20×28 看不清）
    cv::Mat big;
    cv::resize(bin, big, cv::Size(bin.cols * 10, bin.rows * 10), 0, 0, cv::INTER_NEAREST);
    cv::imwrite("debug_digit_input.png", big);

    // ==================== ④ 类别下标 → 装甲板编号 ====================
    if (best == 8)          // 第 8 类是 negative：没识别出数字
    {
        std::cout << "  数字识别失败（判为 negative）  " << out[8] << std::endl;
        return 0;           // 0 表示"未识别"
    }

    int digit = best + 1;   // 下标 = 编号 − 1

    std::cout << "  数字识别: 第 " << best << " 类 → 装甲板编号 " << digit
              << "   [该类=" << out[best] << "  negative=" << out[8] << "]" << std::endl;

    return digit;
}




std::vector<Detection> TraditionalDetector::matchLightBars(const cv::Mat& img, const std::vector<LightBar>& bars)
{

    std::vector<bool> used(bars.size(), false); //灯条是否已经配对
    std::vector<Detection> results; //结果装甲板

    while(true)
    {
        // best_* 是本轮贪心的"当前最优解"。
        // best_cost 初值给 1000：pairCost 不合格时正好返回 1000，
        // 用【严格小于】比较就永远选不到不合格的组合。
        float best_cost = 1000.0f;
        int   best_i    = -1;       // 本轮最佳配对的两根灯条下标
        int   best_j    = -1;       // -1 表示"本轮还没找到任何合格配对"

        // ---------- 遍历所有"两两组合" ----------
        for (size_t i = 0; i < bars.size(); ++i)
        {
            if (used[i]) continue;              // 已被占用的灯条不参与本轮

            // j 从 i+1 开始，有两个好处：
            //   ① 每对只算一次（不会出现 B+A 又把 A+B 算一遍）
            //   ② 天然避免了 i == j（自己和自己配）
            for (size_t j = i + 1; j < bars.size(); ++j)
            {
                if (used[j]) continue;          // 已被占用的灯条不参与本轮

                // 给这一对打分：不合格返回 1000，合格返回 0~1 的小数（具体算法见上方pairCost函数）
                float cost = pairCost(bars[i], bars[j]);

                // 贪心算法
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_i    = static_cast<int>(i);   
                    best_j    = static_cast<int>(j);
                }
            }
        }

        //如果本轮没找到组合，直接遗憾离场
        if(best_i < 0)   break;
        

        //把best_i和best_j配成装甲板
        //分出左右灯条
        bool i_is_left = (bars[best_i].center.x < bars[best_j].center.x);
        const LightBar& left_bar  = i_is_left ? bars[best_i] : bars[best_j];
        const LightBar& right_bar = i_is_left ? bars[best_j] : bars[best_i];

        //拼出装甲板的四个角点（顺序：左上 → 右上 → 右下 → 左下）
        std::vector<cv::Point2f> corners(4);     
        corners[0] = left_bar.top;               // 左上
        corners[1] = right_bar.top;              // 右上
        corners[2] = right_bar.bottom;           // 右下
        corners[3] = left_bar.bottom;            // 左下


        //数字识别：认出这是几号装甲板
        int digit = recognizeDigit(img, left_bar, right_bar);

        //打包成Detection
        Detection det;
        det.kpt   = corners;                    // 四个角点（已是 canonical 顺序）
        det.cls   = digit;                          // 装甲板类型：本数据集神经网络检出的 6 块全是 3，
        det.score = 1.0f;                       // 传统方案没有"置信度"，给 1.0 表示满信
        det.box   = cv::Rect2f(cv::boundingRect(corners));   // 四个角点的外接矩形（画框用）

        results.push_back(det);

        //已经配对的灯条标记为已用
        used[best_i] = true;
        used[best_j] = true;

        // 调试打印（确认没问题后可以注释掉）
        std::cout << "配对: " << best_i << " + " << best_j
                  << "  代价=" << best_cost << std::endl;
    }

    return results;
}



std::vector<Detection> TraditionalDetector::detect(const cv::Mat& img)
{
    cv::Mat mask = binarize(img);
    std::vector<LightBar> bars = findLightBars(mask);
    return matchLightBars(img, bars);        
}



void TraditionalDetector::draw(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
    {
        // ---------- ① 四个角点：Point2f → Point ----------
        // cv::polylines 要求整数坐标，而角点是小数，所以逐个转换。
        // 画图不需要亚像素精度，直接取整即可。
        std::vector<cv::Point> pts;
        for (const auto& p : d.kpt)
            pts.push_back(cv::Point(int(p.x), int(p.y)));

        // ---------- ② 连成闭合四边形 ----------
        // 颜色用黄色 Scalar(0,255,255)（注意 BGR 顺序），
        // 和神经网络的绿框区分开，方便两种方案同框对比。
        if (pts.size() >= 4)
            cv::polylines(img, pts, true, cv::Scalar(0, 255, 255), 3);

        // ---------- ③ 在框的左上角写标签 ----------
        // cv::putText 的 Hershey 字体不支持中文，所以用 "TRAD" 前缀。
        // 传统方案没有置信度（score 恒为 1.0），所以不打印 score。
        std::string text = cv::format("TRAD cls=%d", d.cls);
        cv::putText(img, text,
                    cv::Point(int(d.box.x), int(d.box.y + d.box.height) + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    }
}