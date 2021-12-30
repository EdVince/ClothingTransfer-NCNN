#include "lipjppt.h"

LIPJPPNet::LIPJPPNet()
{
    net.load_param("assert/LIP_JPPNet/model-sim-opt-opt.param");
    net.load_model("assert/LIP_JPPNet/model-sim-opt-opt.bin");
}
LIPJPPNet::~LIPJPPNet()
{
    net.clear();
}

cv::Mat LIPJPPNet::call(cv::Mat rgb)
{
    cv::Mat image;
    cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);


    std::vector<ncnn::Mat> out;
    // 原本方向
    out.push_back(inference(image, int(384 * 1.00), int(384 * 1.00), "tail_output"));
    out.push_back(inference(image, int(384 * 0.75), int(384 * 0.75), "tail_output"));
    out.push_back(inference(image, int(384 * 1.25), int(384 * 1.25), "tail_output"));
    // 翻转方向
    cv::flip(image, image, 1);
    out.push_back(flip(inference(image, int(384 * 1.00), int(384 * 1.00), "tail_output_rev")));
    out.push_back(flip(inference(image, int(384 * 0.75), int(384 * 0.75), "tail_output_rev")));
    out.push_back(flip(inference(image, int(384 * 1.25), int(384 * 1.25), "tail_output_rev")));

    // 融合+argmax
    int N = out.size(), C = out[0].c, H = out[0].h, W = out[0].w;
    std::vector<float> element(C, 0);
    cv::Mat res(H, W, CV_8UC1);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                // 求n个output的平均值
                float sum = 0.f;
                for (int n = 0; n < N; n++) {
                    sum += out[n].channel(c)[h * W + w];
                }
                out[0].channel(c)[h * W + w] = sum / N;
                // 记录最终output
                element[c] = out[0].channel(c)[h * W + w];
            }
            // 求最终output的argmax
            res.at<unsigned char>(h, w) = std::distance(element.begin(), std::max_element(element.begin(), element.end()));
        }
    }

    return res;
}

ncnn::Mat LIPJPPNet::inference(cv::Mat image, int w, int h, std::string node)
{
    ncnn::Mat out_mat;
    {
        ncnn::Mat in_mat = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PixelType::PIXEL_BGR, 256, 256, w, h);
        in_mat.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = net.create_extractor();
        ex.input("image_batch", in_mat);
        ex.extract(node.c_str(), out_mat);
    }
    return out_mat;
}

ncnn::Mat LIPJPPNet::flip(const ncnn::Mat& mask)
{
    int C = mask.c, H = mask.h, W = mask.w;
    ncnn::Mat output(W, H, C);
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output.channel(c)[h * W + w] = mask.channel(c)[h * W + W - w - 1];
            }
        }
    }
    return output;
}