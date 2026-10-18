#include "traditional_detector.hpp"
#include <iostream>     
#include <algorithm>    
#include <cmath>     
#include <memory>   
#include <openvino/openvino.hpp>  
#define ENABLE_DEBUG 0


// 传统方案里所有能调的参数都放在这里。
// 这些数不是随便填的，是拿 test1~test3 三张图和素材视频反复试出来的结果，
// 集中在一起主要是为了方便以后换场景时一处就能改完。
// 每个值为什么取这么大，在用到它的地方写了原因。
namespace {

// 二值化
constexpr int   BIN_DIFF_TH      = 20;    // 通道差（目标通道减去另外两个通道里较大的那个）
constexpr int   BIN_BRIGHT_TH    = 90;    // 目标通道的亮度。这两个条件要同时满足
constexpr int   BIN_CLOSE_KSIZE  = 5;     // 闭运算的核大小。真实相机拍的红灯条中间会过曝成白色，
                                          // 那里通道差反而接近 0，灯条在 mask 里变成空心的，
                                          // 3x3 补不上，所以用 5x5

// 灯条筛选：找到轮廓之后，用下面几条判断它是不是一根灯条
constexpr double BAR_MIN_AREA     = 12.0;   // 轮廓面积下限，太小的碎片直接跳过
constexpr float  BAR_MIN_ASPECT   = 1.5f;   // 长宽比下限，灯条是细长的
constexpr float  BAR_MIN_LENGTH   = 6.0f;   // 长边的像素长度下限
constexpr float  BAR_MIN_FILL     = 0.4f;   // 填充率下限（轮廓面积 / 外接旋转矩形面积），
                                            // 用来排除"细长但是弯弯曲曲"的东西
constexpr float  BAR_MIN_WIDTH    = 2.0f;   // 短边宽度下限
constexpr float  BAR_MAX_TILT_DEG = 35.0f;  // 长轴和竖直方向的最大夹角。
                                            // 这条是关键：地上的标线也是细长的，长宽比甚至能到 20 以上，
                                            // 只有靠"横的还是竖的"才能把它和灯条分开

// 灯条配对：判断两根灯条是不是同一块装甲板的左右两边
constexpr float PAIR_MAX_LEN_RATIO    = 1.5f;   // 两根灯条的长度比上限。
                                                // 主要是为了排除"一根灯条断成两截"产生的碎片
constexpr float PAIR_MIN_DIST_RATIO   = 1.5f;   // 中心距 / 平均灯条长 的下限
constexpr float PAIR_MAX_DIST_RATIO   = 3.6f;   // 上限。理论上这个比值应该等于装甲板宽高比
                                                // 135/55 ≈ 2.45，但板子斜着的时候宽度被压缩了，
                                                // 实测三块真板只有 1.73 / 1.84 / 2.10，
                                                // 所以区间必须放宽，不然一块都配不出来
constexpr float PAIR_MAX_ANGLE_DIFF   = 12.0f;  // 同一块板的两根灯条几乎平行，实测倾角差都在 3 度以内
constexpr float PAIR_MAX_DY_RATIO     = 0.5f;   // 两根灯条中心的高度差 / 平均灯条长
constexpr float PAIR_IDEAL_DIST_RATIO = 2.45f;  // 理想宽高比，就是 135/55
constexpr float PAIR_INVALID_COST     = 1000.0f;// "这一对不合格"的返回值。故意给一个比任何合格代价都大的数，
                                                // 这样调用方只要比大小就够了，不用另外判断合不合格

// 数字识别
constexpr float DIGIT_ROI_H_SCALE   = 1.9f;   // 裁剪高度 = 灯条长 × 这个系数。
                                              // 板上印的数字比灯条还高，按灯条长截会把数字的上下切掉
constexpr float DIGIT_ROI_MAX_W     = 0.6f;   // 裁剪宽度上限 = 两根灯条的间距 × 这个系数。
                                              // 灯条是画面里最亮的东西，被框进来会盖住数字
constexpr int   DIGIT_INPUT_W       = 20;     // 数字网络的输入宽
constexpr int   DIGIT_INPUT_H       = 28;     // 数字网络的输入高
constexpr int   DIGIT_NEGATIVE_CLS  = 8;      // 网络第 8 类输出是 negative，意思是"这里不是数字"
constexpr int   DIGIT_NUM_CLASSES   = 9;      // 数字网络一共 9 个输出
constexpr int   DIGIT_INDEX_OFFSET  = 1;      // 网络输出的下标比装甲板编号小 1（编号 3 对应第 2 类）
constexpr int   DIGIT_ROI_MIN_SIZE  = 4;      // 裁剪区域的最小边长，比这个还小就认不出来了

// 画图
constexpr int    TRAD_LINE_THICK = 3;      // 传统方案的框线宽，比神经网络的粗，两张框叠在一起时能分开
constexpr double TEXT_FONT_SCALE = 0.5;
constexpr int    TEXT_OFFSET_PX  = 18;     // 标签相对框的偏移。putText 的 y 指的是文字基线不是顶部，
                                           // 字号 0.5 时字高大概 15 像素，所以取 18

} // namespace



TraditionalDetector::TraditionalDetector(bool is_blue, const std::string& digit_model_path)
    : is_blue_(is_blue)
{
    // 数字网络只加载一次。千万别放到 detect 里面去，
    // 不然处理视频的时候每帧都要重新编译一次模型，会慢到没法用
    std::cout << "正在加载数字网络: " << digit_model_path << std::endl;

    std::shared_ptr<ov::Model> model = digit_core_.read_model(digit_model_path);
    digit_model_   = digit_core_.compile_model(model, "CPU");
    digit_request_ = digit_model_.create_infer_request();
}

// 把原图变成灯条的 mask。
// 这里用的是"颜色差"而不是单纯看亮度：灯条本身有颜色（红方红、蓝方蓝），
// 而场地、白色数字、纸箱这些东西 R/G/B 三个通道差不多亮，减完接近 0，一下子就被排除了。
cv::Mat TraditionalDetector::binarize(const cv::Mat& img)
{
    //分离R G B三通道
    std::vector<cv::Mat> ch;
    cv::split(img, ch);   // ch[0]=B, ch[1]=G, ch[2]=R


    //算出"目标通道-另外两个通道的最大值"
    // 为什么要取另两个的 max 再减：灯条的相邻通道不一定完全是 0，
    // 比如红灯条的 G 通道也有一定亮度，减掉较大的那个才能保证"只有纯色才留下"
    // subtract 是饱和减法，减出负数会自动变 0，所以不用担心下溢
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

    #if ENABLE_DEBUG
        cv::imwrite("debug_score.jpg", score); // 调试用：看每根灯条到底有多"蓝"
    #endif     
    //两个条件同时满足才算灯条像素
    cv::Mat m1, m2, mask;
    cv::threshold(score, m1, BIN_DIFF_TH, 255, cv::THRESH_BINARY);                  // 通道差 > 20
    cv::threshold(ch[is_blue_ ? 0 : 2], m2, BIN_BRIGHT_TH, 255, cv::THRESH_BINARY);   // 亮度 > 90
    cv::bitwise_and(m1, m2, mask);

    //形态学闭运算：把灯条亮芯的断裂连成一整块
    // 顺带也能把过曝造成的空心填实（所以要 5x5，见上面参数说明）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(BIN_CLOSE_KSIZE, BIN_CLOSE_KSIZE));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    #if ENABLE_DEBUG
        // 临时诊断：定阈值用（调好后删）
        double s_min, s_max, r_min, r_max;
        cv::minMaxLoc(score, &s_min, &s_max);
        cv::minMaxLoc(ch[is_blue_ ? 0 : 2], &r_min, &r_max);
        std::printf("  [二值化] 通道差 %3.0f~%3.0f  目标通道 %3.0f~%3.0f  白色占比 %.2f%%\n",
                    s_min, s_max, r_min, r_max,
                    100.0 * cv::countNonZero(mask) / mask.total());
    #endif
    
    return mask;
}



// 从 mask 里找出所有灯条。
// 每找到一个轮廓，就先取它的最小外接旋转矩形，再用长宽比、填充率、倾角这几条判断它像不像灯条。
// 注意必须用旋转矩形而不是普通的 boundingRect：灯条是斜的，轴对齐的框会把大量背景框进来，
// 量出来的长和宽根本不是灯条的实际尺寸。
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
        // 先按面积粗筛一遍，能省掉后面大量没必要的最小矩形计算
        double area = cv::contourArea(contours[i]);

        if(area < BAR_MIN_AREA)
            continue;
        //画出最小矩形
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        //取矩形的四个端点
        // 用 rect.points() 而不是 cv::boxPoints：boxPoints 内部要往一个 4x2 的矩阵里写，
        // 塞不进 std::vector，用会直接抛异常，points() 填定长数组就没这个问题
        cv::Point2f rect_pts[4];
        rect.points(rect_pts);

        //算出相邻两条边的长度
        // 只算相邻的两条边就够了（一条长一条短），
        float e1 = cv::norm(rect_pts[1] - rect_pts[0]);
        float e2 = cv::norm(rect_pts[2] - rect_pts[1]);

        //比较这两条边哪条长，哪条短，长的存入length，短的存入width
        float length = std::max(e1,e2);
        float width  = std::min(e1,e2);

        //得到"长边向量"
        // 这个向量后面要用来算倾角，所以方向要跟着哪条边长走
        cv::Point2f edge; 

        if(e1>e2)
            edge = rect_pts[1] - rect_pts[0]; 
        else
            edge = rect_pts[2] - rect_pts[1]; 
        
        //算出夹角
        // atan2(y, x) 的参数顺序是先 y 后 x，不要写反。
        // 这里要的是"长轴和竖直方向的夹角"，也就是 水平分量/垂直分量 的反正切，
        // 所以写成 atan2(|dx|, |dy|)。取了绝对值以后结果一定落在 0~90 度，
        // 不用再管方向是朝上还是朝下
        float tilt = std::atan2(std::abs(edge.x), std::abs(edge.y)) * 180.0 / CV_PI;

        //算出填充率（同时排除除以0的情况）
        // 万一边长算出来是 0（退化轮廓），下面的除法会得到 inf，后面比较就全乱了
        if(length * width == 0) continue;

        double fill_rate = area / (length * width);

        // 五条判据必须全部通过
        if(length/width <= BAR_MIN_ASPECT || length < BAR_MIN_LENGTH || 
            fill_rate <= BAR_MIN_FILL || tilt >= BAR_MAX_TILT_DEG || width < BAR_MIN_WIDTH) continue;

        //定义两个角点，上角点和下角点
        // 讲义里说的"取对称轴的两个端点"，其实就是最小矩形两条【短边】的中点。
        // 所以先看哪两条边是短边：e1 比 e2 长的话，短边就是 (1-2) 和 (3-0)
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
        // 图像坐标的 y 是往下增大的，所以 y 小的那个才是上面。
        // 这一步必须做，不然配对的时候四个角点的顺序会乱，PnP 解出来的位姿就错了
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


    #if ENABLE_DEBUG
        // 每帧都打印会刷屏（视频有 531 帧），所以只在调试模式下输出
        std::cout << "筛出灯条: " << bars.size() << " 根" << std::endl;
    #endif

    return bars;
}
 


// 给一对灯条打分：合格就返回一个 0~1 的小数，越小说明这一对越像真正的装甲板；
// 不合格直接返回 PAIR_INVALID_COST（一个很大的数）。
// 之所以用"分数"而不是简单判个合格/不合格，是因为有的组合虽然都合格但明显有更好的，
// 比如一块板上真正的那两根，和"左边板的右灯条 + 右边板的左灯条"这种跨板的组合，
// 两者都能通过判据，要靠分数才能选出前者。
static float pairCost(const LightBar& a, const LightBar& b) //给一对灯条打分
{
    //退化的灯条直接判不合格
    if (a.length <= 0.0f || b.length <= 0.0f)
        return PAIR_INVALID_COST;

    // 打分要用到的三个量
    //平均灯条长
    // 用两根的平均值当"尺子"，比只用其中一根公平一些
    float avg_len = (a.length + b.length) / 2.0f;

    //灯条中心距
    float dist = cv::norm(a.center - b.center);

    // 距离比 = 中心距 ÷ 平均灯条长度
    float len_ratio = dist / avg_len;

    // 下面是四条判据

    //判据1：两根灯条长度比不能过大
    // 同一块板的两根灯条一样长。如果差太多，多半是一根灯条断裂后留下的小碎片
    float longer  = std::max(a.length, b.length);
    float shorter = std::min(a.length, b.length);
    if (longer / shorter >= PAIR_MAX_LEN_RATIO)
        return PAIR_INVALID_COST;

    //判据2：距离比要落在合理区间
    if (len_ratio < PAIR_MIN_DIST_RATIO || len_ratio > PAIR_MAX_DIST_RATIO)
        return PAIR_INVALID_COST;

    //判据3：两根灯条要近似平行
    if (std::abs(a.angle - b.angle) >= PAIR_MAX_ANGLE_DIFF)
        return PAIR_INVALID_COST;

    //判据4：两根灯条的高度要接近（两根灯条中心的纵坐标差应该远小于灯条长度）
    if (std::abs(a.center.y - b.center.y) / avg_len >= PAIR_MAX_DY_RATIO)
        return PAIR_INVALID_COST;

    //上述4个判据全部通过后，返回成绩（代价）：数字越小越像一个装甲板
    // 用离理想宽高比 2.45 的相对偏差当分数
    return std::abs(len_ratio - PAIR_IDEAL_DIST_RATIO) / PAIR_IDEAL_DIST_RATIO;
}



// 认出这块装甲板是几号。做法是把两根灯条中间那块区域抠出来，
// 摆正、缩放、二值化之后喂给 mlp.onnx，取分数最高的那一类。
// 返回装甲板编号（1~9）；如果网络判断"这里不是数字"就返回 0。
int TraditionalDetector::recognizeDigit(const cv::Mat& img,
                                       const LightBar& left,
                                       const LightBar& right)
{
    // 先确定裁剪区域的位置和大小
    // 数字就在两根灯条中间，所以取它们中心连线的中点
    cv::Point2f center((left.center.x + right.center.x) / 2.0f,
                       (left.center.y + right.center.y) / 2.0f);

    float bar_dist = cv::norm(left.center - right.center);
    float avg_len  = (left.length + right.length) / 2.0f;

    // 两灯条中心连线的倾角 = 装甲板的倾斜角
    float deg = std::atan2(right.center.y - left.center.y,
                           right.center.x - left.center.x) * 180.0 / CV_PI;

    // 绕 center 旋转把板摆平（绕 center 转 → center 位置不变，可继续用它截取）
    // 为什么一定要先转正：板子在图里是斜的（实测 9~14 度），轴对齐的裁剪框要盖住斜板，
    // 高度必须等于板的外接矩形高度，比板本身高大约 10 像素，
    // 直接按板高裁就会把数字的上下切掉
    cv::Mat rot = cv::getRotationMatrix2D(center, deg, 1.0);
    cv::Mat upright;
    cv::warpAffine(img, upright, rot, img.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // 高度 = 灯条长 × 系数（数字比灯条高，必须留余量，否则上下被切）
    int roi_h = int(avg_len * DIGIT_ROI_H_SCALE);

    // 宽度按目标比例推（28:20 = 1.4），再限制不超过灯条间距的 0.6 倍（避免框进灯条）
    // 这里"宁可有一点变形也不能让灯条进框"，因为灯条比数字亮得多，进来就把数字盖住了。
    // 而配对判据保证了中心距 ≥ 1.5 倍灯条长，所以按 0.6 倍限宽是安全的
    int roi_w = int(roi_h * float(DIGIT_INPUT_H) / float(DIGIT_INPUT_W));
    if (roi_w > int(bar_dist * DIGIT_ROI_MAX_W)) roi_w = int(bar_dist * DIGIT_ROI_MAX_W);

    if (roi_w < DIGIT_ROI_MIN_SIZE || roi_h < DIGIT_ROI_MIN_SIZE) return 0;

    cv::Mat roi;
    cv::getRectSubPix(upright, cv::Size(roi_w, roi_h), center, roi);
    if (roi.empty()) return 0;

    // 灰度 → 缩放 → Otsu 二值化 → 归一化
    // 二值化这一步是关键：这个网络要的是"黑白数字图"，喂灰度图会全部判成 negative。
    // 这是我把尺寸和归一化组合扫了一遍（2 种方向 × 5 种归一化）才试出来的，
    // 10 种组合里只有"Otsu 二值化 + 这个尺寸方向"能正确认出数字
    cv::Mat gray, resized, bin, blob;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(DIGIT_INPUT_W, DIGIT_INPUT_H));
    cv::threshold(resized, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    bin.convertTo(blob, CV_32F, 1.0 / 255.0);

    // 推理
    // 张量形状固定写成 {1,20,28,1}，这是模型的输入规格，不能改
    ov::Tensor input(ov::element::f32, {1, DIGIT_INPUT_W, DIGIT_INPUT_H, 1}, blob.ptr<float>());
    digit_request_.set_input_tensor(input);
    digit_request_.infer();
    const float* out = digit_request_.get_output_tensor().data<const float>();

    // 在 9 个输出里找分数最高的那个
    int best = 0;
    for (int k = 1; k < DIGIT_NUM_CLASSES; ++k) if (out[k] > out[best]) best = k;
    

    #if ENABLE_DEBUG
        // 调试：把二值化后的输入存下来（放大 10 倍，不然 20×28 看不清）
        cv::Mat big;
        cv::resize(bin, big, cv::Size(bin.cols * 10, bin.rows * 10), 0, 0, cv::INTER_NEAREST);
        cv::imwrite("debug_digit_input.png", big);
    #endif
    
        // 类别下标 → 装甲板编号
        // 第 8 类是 negative，也就是"这块区域里没有数字"。
        // 真实场景里人的胳膊、红衣服、背景反光经常被误配成一对灯条，
        // 它们中间没有数字，就是靠这一步剔除的
        if (best == DIGIT_NEGATIVE_CLS)
        {
        #if ENABLE_DEBUG
            std::cout << "  数字识别失败（判为 negative）  " << out[8] << std::endl;
        #endif
            return 0;           // 0 表示"未识别"
        }
        // 网络输出的下标比装甲板编号小 1：编号 3 对应第 2 类。
        // 这个对应关系是用两块板对出来的（数字 3 输出第 2 类、数字 4 输出第 3 类），
        // 而装甲板网络那边的类别号是直接等于编号的，两个网络的编号规则不一样，用的时候要分开记
        int digit = best + DIGIT_INDEX_OFFSET;
        
    #if ENABLE_DEBUG
        std::cout << "  数字识别: 第 " << best << " 类 → 装甲板编号 " << digit
                << "   [该类=" << out[best] << "  negative=" << out[8] << "]" << std::endl;
    #endif


    return digit;
}




// 把灯条两两配对，配上的就作为一块装甲板输出。
// 用的是"贪心"：每一轮从还没被用过的灯条里，挑分数最好的一对，
// 选中之后把这两根灯条标记成已用，再开始下一轮，直到再也找不到合格的组合。
// 为什么需要"贪心 + 标记已用"：一根灯条只能属于一块装甲板，
// 如果不做标记，A+B 和 B+C 都合格的话 B 就会被用两次，输出两块互相矛盾的板
std::vector<Detection> TraditionalDetector::matchLightBars(const cv::Mat& img, const std::vector<LightBar>& bars)
{

    std::vector<bool> used(bars.size(), false); //灯条是否已经配对
    std::vector<Detection> results; //结果装甲板

    while(true)
    {
        // best_* 是本轮贪心的"当前最优解"。
        // best_cost 初值给 1000：pairCost 不合格时正好返回 1000，
        // 用【严格小于】比较就永远选不到不合格的组合。
        float best_cost = PAIR_INVALID_COST;
        int   best_i    = -1;       // 本轮最佳配对的两根灯条下标
        int   best_j    = -1;       // -1 表示"本轮还没找到任何合格配对"

        // 遍历所有两两组合
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
        // 这种情况会出现在灯条是奇数根的时候，最后一根配不上，很正常
        if(best_i < 0)   break;
        

        //把best_i和best_j配成装甲板
        //分出左右灯条
        // 中心 x 小的那根就是左边。这一步必须做，不然四个角点的左右顺序会乱
        bool i_is_left = (bars[best_i].center.x < bars[best_j].center.x);
        const LightBar& left_bar  = i_is_left ? bars[best_i] : bars[best_j];
        const LightBar& right_bar = i_is_left ? bars[best_j] : bars[best_i];

        //拼出装甲板的四个角点（顺序：左上 → 右上 → 右下 → 左下）
        // 这个顺序必须和 pnp_solver 里 obj_pts 的顺序、types.hpp 里的说明完全一致，
        // 中间任何一个地方对不上，解出来的位姿就是错的（而且不会报错）
        std::vector<cv::Point2f> corners(4);     
        corners[0] = left_bar.top;               // 左上
        corners[1] = right_bar.top;              // 右上
        corners[2] = right_bar.bottom;           // 右下
        corners[3] = left_bar.bottom;            // 左下


        //数字识别：认出这是几号装甲板
        int digit = recognizeDigit(img, left_bar, right_bar);

        // 如果开启了"必须有数字"这个选项，认不出数字的就不再当成装甲板
        // 注意这里必须把这两根灯条标记成已用，不然下一轮 while 还会选到同一对，
        // 然后又被 continue 掉，就死循环了
        if (require_digit_ && digit == 0)
        {
            used[best_i] = true;
            used[best_j] = true;
            continue;
        }

        //打包成Detection
        Detection det;
        det.kpt   = corners;                    // 四个角点（已是 canonical 顺序）
        det.cls   = digit;                      // 装甲板编号（由 mlp.onnx 识别；0 = 未识别出数字）
        det.score = 1.0f;                       // 传统方案没有"置信度"这个概念，给 1.0 表示满信
        det.box   = cv::Rect2f(cv::boundingRect(corners));   // 四个角点的外接矩形（画框用）

        results.push_back(det);

        //已经配对的灯条标记为已用
        used[best_i] = true;
        used[best_j] = true;


        #if ENABLE_DEBUG
            std::cout << "配对: " << best_i << " + " << best_j
                    << "  代价=" << best_cost << std::endl;
        #endif
    }

    return results;
}



// 传统方案的入口，三步走：二值化 → 找灯条 → 配对
std::vector<Detection> TraditionalDetector::detect(const cv::Mat& img)
{
    cv::Mat mask = binarize(img);
    std::vector<LightBar> bars = findLightBars(mask);
    return matchLightBars(img, bars);        
}



// 把识别结果画在图上。接口和 YoloDetector::draw 一样，
// 因为两种方案的输出类型和角点顺序都是统一的，只有颜色和标签不一样。
void TraditionalDetector::draw(cv::Mat& img, const std::vector<Detection>& dets)
{
    for (const auto& d : dets)
    {
        // 四个角点转成整数坐标：polylines 要整数，画图也不需要亚像素精度
        std::vector<cv::Point> pts;
        for (const auto& p : d.kpt)
            pts.push_back(cv::Point(int(p.x), int(p.y)));

        // 连成闭合四边形。颜色用黄色，和神经网络的绿框区分开，方便两种方案同框对比。
        // 注意 OpenCV 的颜色是 BGR 顺序，Scalar(0,255,255) 才是黄色
        if (pts.size() >= 4)
            cv::polylines(img, pts, true, cv::Scalar(0, 255, 255), TRAD_LINE_THICK);

        // 写标签。putText 用的是 Hershey 字体，只支持 ASCII，写中文会变成问号，所以用 TRAD 前缀。
        // 传统方案没有置信度（score 恒为 1.0），就不打印了。
        // 文字放在框的【下方】，因为神经网络方案的标签在框上方，一上一下就不会重叠
        std::string text = cv::format("TRAD cls=%d", d.cls);
        cv::putText(img, text,
                    cv::Point(int(d.box.x), int(d.box.y + d.box.height) + TEXT_OFFSET_PX),
                    cv::FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, cv::Scalar(0, 255, 255), 1);
    }
}