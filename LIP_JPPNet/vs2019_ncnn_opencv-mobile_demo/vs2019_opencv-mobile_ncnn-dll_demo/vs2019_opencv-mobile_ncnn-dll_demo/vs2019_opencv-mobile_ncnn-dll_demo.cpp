#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>

class LIP_JPPNet
{
private:
    ncnn::Net net;
    const float mean_vals[3] = { 104.00698793f, 116.66876762f, 122.67891434f };
    const float norm_vals[3] = { 1.f, 1.f, 1.f };
    const int label_colours[20][3] = { {0, 0, 0}, {128, 0, 0}, {255, 0, 0}, {0, 85, 0}, {170, 0, 51}, {255, 85, 0}, {0, 0, 85}, {0, 119, 221}, {85, 85, 0}, {0, 85, 85}, {85, 51, 0}, {52, 86, 128}, {0, 128, 0}, {0, 0, 255}, {51, 170, 221}, {0, 255, 255}, {85, 255, 170}, {170, 255, 85}, {255, 255, 0}, {255, 170, 0} };

public:
    LIP_JPPNet() {
        net.load_param("assert/model-sim-opt-opt.param");
        net.load_model("assert/model-sim-opt-opt.bin");
    }
    ~LIP_JPPNet() {
        net.clear();
    }
    ncnn::Mat inference(cv::Mat image, int w, int h, std::string node) 
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
    cv::Mat call(cv::Mat image)
    {
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
    ncnn::Mat flip(const ncnn::Mat& mask)
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
    cv::Mat show(cv::Mat mask)
    {
        int H = mask.rows, W = mask.cols;
        cv::Mat output(H, W, CV_8UC3);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = mask.at<unsigned char>(h, w);
                output.at<cv::Vec3b>(h, w)[0] = label_colours[idx][0];
                output.at<cv::Vec3b>(h, w)[1] = label_colours[idx][1];
                output.at<cv::Vec3b>(h, w)[2] = label_colours[idx][2];
            }
        }
        return output;
    }
};

int main()
{
    LIP_JPPNet lip_jpp;

    cv::Mat image = cv::imread("assert/06_1_front.jpg", CV_LOAD_IMAGE_COLOR);

    cv::Mat mask = lip_jpp.call(image);

    cv::imwrite("mask.png", mask);

    cv::Mat show = lip_jpp.show(mask);
    cv::cvtColor(show, show, cv::COLOR_RGB2BGR);
    cv::imwrite("mask_vis.png", show);


    return 0;
}
