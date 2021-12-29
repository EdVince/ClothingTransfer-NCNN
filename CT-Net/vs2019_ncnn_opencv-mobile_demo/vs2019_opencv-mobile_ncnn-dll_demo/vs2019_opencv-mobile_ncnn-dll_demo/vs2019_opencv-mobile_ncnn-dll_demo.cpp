#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

void show(cv::Mat in)
{
    std::cout << in.size() << ", sum:" << cv::sum(cv::sum(in))[0] << std::endl;
}

void show(ncnn::Mat in)
{
    double sum = 0;
    for (int c = 0; c < in.c; c++) {
        for (int hw = 0; hw < in.h * in.w; hw++) {
            sum += in.channel(c)[hw];
        }
    }
    std::cout << "(" << in.c << "," << in.h << "," << in.w << "), sum:" << sum << std::endl;
}

ncnn::Mat get_label_tensor(cv::Mat candidate, cv::Mat subset, cv::Mat img)
{
    const int limbSeq[][2] = { {2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17}, {1, 16}, {16, 18}, {3, 17}, {6, 18} };
    const int colors[][3] = { {255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85} };

    cv::Mat canvas = cv::Mat::zeros(img.size(), CV_8UC3);

    for (int i = 0; i < 18; i++) {
        int index = subset.at<float>(i, 0);
        if (index == -1) continue;
        int x = candidate.at<float>(index, 0), y = candidate.at<float>(index, 1);
        cv::circle(canvas, cv::Point(x, y), 4, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), -1);
    }

    std::vector<cv::Mat> joints;
    for (int i = 0; i < 17; i++) {
        float indexA = subset.at<float>(limbSeq[i][0] - 1, 0), indexB = subset.at<float>(limbSeq[i][1] - 1, 0);
        cv::Mat cur_canvas = canvas.clone();
        if (indexA == -1 || indexB == -1) {
            joints.push_back(cv::Mat::zeros(cur_canvas.size(), CV_8UC1));
            continue;
        }
        float Y0 = candidate.at<float>(indexA, 0), Y1 = candidate.at<float>(indexB, 0);
        float X0 = candidate.at<float>(indexA, 1), X1 = candidate.at<float>(indexB, 1);
        float mX = (X0 + X1) / 2.0f;
        float mY = (Y0 + Y1) / 2.0f;
        float length = std::sqrt(std::pow(X0 - X1, 2) + std::pow(Y0 - Y1, 2));
        float angle = std::atan2(X0 - X1, Y0 - Y1) * 45.0 / std::atan(1.0);
        std::vector<cv::Point> polygon;
        cv::ellipse2Poly(cv::Point(int(mY), int(mX)), cv::Size(int(length / 2), 1), int(angle), 0, 360, 1, polygon);
        cv::fillConvexPoly(cur_canvas, polygon, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]));
        cv::addWeighted(canvas, 0.4, cur_canvas, 0.6, 0, canvas);
        cv::Mat joint = cv::Mat::zeros(cur_canvas.size(), CV_8UC1);
        cv::fillConvexPoly(joint, polygon, 255);
        cv::addWeighted(joint, 0.4, joint, 0.6, 0, joint);
        joints.push_back(joint);
    }

    

    const float norm1[3] = { 1.0f / 255.0f };
    const float norm3[3] = { 1.0f / 255.0f,1.0f / 255.0f ,1.0f / 255.0f };

    ncnn::Mat label_tensor(256, 256, 20);

    cv::Mat pose;
    cv::cvtColor(canvas, pose, cv::COLOR_BGR2RGB);
    cv::resize(pose, pose, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    ncnn::Mat tensor_pose = ncnn::Mat::from_pixels(pose.data, ncnn::Mat::PIXEL_RGB, pose.cols, pose.rows);
    tensor_pose.substract_mean_normalize(0, norm3);

    for (int c = 0; c < tensor_pose.c; c++)
        for (int hw = 0; hw < tensor_pose.h * tensor_pose.w; hw++)
            label_tensor.channel(c)[hw] = tensor_pose.channel(c)[hw];

    for (int i = 0; i < joints.size(); i++) {
        cv::Mat im_dist;
        cv::distanceTransform(255 - joints[i], im_dist, cv::DIST_L2, 3);
        im_dist.convertTo(im_dist, CV_8UC1); // 不用clip了，转uchar自带clip特效
        ncnn::Mat tensor_dist = ncnn::Mat::from_pixels(im_dist.data, ncnn::Mat::PIXEL_GRAY, im_dist.cols, im_dist.rows);
        tensor_dist.substract_mean_normalize(0, norm1);
        for (int hw = 0; hw < tensor_pose.h * tensor_pose.w; hw++)
            label_tensor.channel(i+3)[hw] = tensor_dist.channel(0)[hw];
    }

    return label_tensor;
}

cv::Mat convert_seg(const cv::Mat& t_seg)
{
    const std::vector<std::vector<int>> mask = { {1, 2, 4, 11, 13}, {3, 14, 15}, {16, 17}, {5, 6, 7, 10}, {9, 12}, {8, 18, 19} };
    cv::Mat c_seg = cv::Mat::zeros(t_seg.size(), CV_32FC1);
    for (int i = 0; i < mask.size(); i++) {
        std::vector<int> items = mask[i];
        for (int& item : items) {
            cv::Mat idx = t_seg == item;
            std::vector<cv::Point> loc;
            cv::findNonZero(idx, loc);
            for (auto& l : loc) {
                c_seg.at<float>(l) = i + 1;
            }
        }
    }
    return c_seg;
}

cv::Mat convert_dp_mask(const cv::Mat& dp)
{
    const std::vector<std::vector<int>> mask = { {23, 24}, {3, 4, 15, 16, 17, 18, 19, 20, 21, 22}, {7, 8, 9, 10, 11, 12, 13, 14}, {1, 2}, {}, {5, 6} };
    cv::Mat channel[3];
    cv::split(dp, channel);
    cv::Mat dp_mask = channel[2];
    cv::Mat c_dp_mask = cv::Mat::zeros(dp_mask.size(), CV_32FC1);
    for (int i = 0; i < mask.size(); i++) {
        std::vector<int> items = mask[i];
        for (int& item : items) {
            cv::Mat idx = dp_mask == item;
            std::vector<cv::Point> loc;
            cv::findNonZero(idx, loc);
            for (auto& l : loc) {
                c_dp_mask.at<float>(l) = i + 1;
            }
        }
    }
    return c_dp_mask;
}

void dataset(ncnn::Mat& input_semantics, ncnn::Mat& ref_semantics, cv::Mat& ref_image, cv::Mat& real_image, cv::Mat& seg_img, cv::Mat& ref_seg, cv::Mat& img_dp_mask)
{
    // src
    cv::Mat image_tensor = cv::imread("assert/person_src/01_7_additional.jpg");
    cv::cvtColor(image_tensor, image_tensor, cv::COLOR_BGR2RGB);
    image_tensor.convertTo(image_tensor, CV_32FC3, 1.0 / 0.5 / 255.0, -1.0);

    ncnn::Mat label_tensor;
    {
        cv::Mat candidate(17, 4, CV_32FC1);
        std::ifstream candidate_in("assert/person_src/01_7_additional_candidate.data", std::ios::in | std::ios::binary);
        candidate_in.read((char*)candidate.data, sizeof(float) * candidate.cols * candidate.rows);

        cv::Mat subset(20, 1, CV_32FC1);
        std::ifstream subset_in("assert/person_src/01_7_additional_subset.data", std::ios::in | std::ios::binary);
        subset_in.read((char*)subset.data, sizeof(float) * subset.cols * subset.rows);

        cv::Mat img = cv::imread("assert/person_src/01_7_additional.jpg");

       label_tensor = get_label_tensor(candidate, subset, img);
    }

    cv::Mat seg_img_tensor = cv::imread("assert/person_src/01_7_additional_seg.png", 0);
    seg_img_tensor.convertTo(seg_img_tensor, CV_32FC1);
    seg_img_tensor = convert_seg(seg_img_tensor);

    cv::Mat img_dp_tensor = cv::imread("assert/person_src/01_7_additional_IUV.png");
    cv::cvtColor(img_dp_tensor, img_dp_tensor, cv::COLOR_BGR2RGB);
    img_dp_tensor.convertTo(img_dp_tensor, CV_32FC3);
    img_dp_tensor = convert_dp_mask(img_dp_tensor);


    // dst
    cv::Mat ref_tensor = cv::imread("assert/person_dst/06_1_front.jpg");
    cv::cvtColor(ref_tensor, ref_tensor, cv::COLOR_BGR2RGB);
    ref_tensor.convertTo(ref_tensor, CV_32FC3, 1.0 / 0.5 / 255.0, -1.0);

    ncnn::Mat label_ref_tensor;
    {
        cv::Mat candidate(17, 4, CV_32FC1);
        std::ifstream candidate_in("assert/person_dst/06_1_front_candidate.data", std::ios::in | std::ios::binary);
        candidate_in.read((char*)candidate.data, sizeof(float) * candidate.cols * candidate.rows);

        cv::Mat subset(20, 1, CV_32FC1);
        std::ifstream subset_in("assert/person_dst/06_1_front_subset.data", std::ios::in | std::ios::binary);
        subset_in.read((char*)subset.data, sizeof(float) * subset.cols * subset.rows);

        cv::Mat img = cv::imread("assert/person_dst/06_1_front.jpg");

        label_ref_tensor = get_label_tensor(candidate, subset, img);
    }

    cv::Mat ref_seg_tensor = cv::imread("assert/person_dst/06_1_front_seg.png", 0);
    ref_seg_tensor.convertTo(ref_seg_tensor, CV_32FC1);
    ref_seg_tensor = convert_seg(ref_seg_tensor);


    // assign
    input_semantics = label_tensor;
    ref_semantics = label_ref_tensor;
    ref_image = ref_tensor;
    real_image = image_tensor;
    seg_img = seg_img_tensor;
    ref_seg = ref_seg_tensor;
    img_dp_mask = img_dp_tensor;
}

ncnn::Mat mask_convert(const cv::Mat& seg, int c)
{
    ncnn::Mat c_seg_tensor(seg.cols, seg.rows, c); c_seg_tensor.fill(0.0f);
    for (int h = 0; h < seg.rows; h++)
        for (int w = 0; w < seg.cols; w++)
            c_seg_tensor.channel(seg.at<float>(h, w))[h * seg.cols + w] = 1;
    return c_seg_tensor;
}

float im2col_get_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

ncnn::Mat im2col_cpu(ncnn::Mat& data_im, int ksize, int stride, int pad)
{
    int channels = data_im.c;
    int height = data_im.h;
    int width = data_im.w;
    data_im = data_im.reshape(width * height * channels, 1, 1); // flatten方便取数据

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;;
    int channels_col = channels * ksize * ksize;

    ncnn::Mat data_col = ncnn::Mat(channels_col * height_col * width_col, 1, 1);
    data_col.fill(0.0f);

    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col.channel(0)[col_index] = im2col_get_pixel(data_im.channel(0), height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }

    data_im = data_im.reshape(width, height, channels); // 还原shape避免修改数据

    return data_col.reshape(height_col * width_col, channels_col);
}

ncnn::Mat cv2ncnn(const cv::Mat& in)
{
    int H = in.rows, W = in.cols;
    if (in.type() == 5) { // CV_32FC1
        ncnn::Mat out(W, H, 1);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.channel(0)[h * W + w] = in.at<float>(h, w);
            }
        }
        return out;
    }
    else if (in.type() == 21) { // CV_32FC3
        ncnn::Mat out(W, H, 3);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.channel(0)[h * W + w] = in.at<cv::Vec3f>(h, w)[0];
                out.channel(1)[h * W + w] = in.at<cv::Vec3f>(h, w)[1];
                out.channel(2)[h * W + w] = in.at<cv::Vec3f>(h, w)[2];
            }
        }
        return out;
    }
    else {
        std::cout << "ncnn2cv cv::Mat::type not support!";
        ncnn::Mat out(W, H, 1);
        out.fill(0.0f);
        return out;
    }
}

cv::Mat ncnn2cv(const ncnn::Mat& in)
{
    int H = in.h, W = in.w;
    if (in.c == 1) {
        cv::Mat out(H, W, CV_32FC1);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.at<float>(h, w) = in.channel(0)[h * W + w];
            }
        }
        return out;
    }
    else if (in.c == 3) {
        cv::Mat out(H, W, CV_32FC3);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                out.at<cv::Vec3f>(h, w)[0] = in.channel(0)[h * W + w];
                out.at<cv::Vec3f>(h, w)[1] = in.channel(1)[h * W + w];
                out.at<cv::Vec3f>(h, w)[2] = in.channel(2)[h * W + w];
            }
        }
        return out;
    }
    else {
        std::cout << "ncnn2cv error, not support!" << std::endl;
        return cv::Mat::zeros(0, 0, CV_32FC1);
    }
}

void col2im_add_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

ncnn::Mat col2im_cpu(ncnn::Mat& data_col, int output_size, int ksize, int stride, int pad)
{
    int data_col_c = data_col.c;
    int data_col_h = data_col.h;
    int data_col_w = data_col.w;

    int height = output_size;
    int width = output_size;
    int channels = int(data_col.h / ksize / ksize);

    ncnn::Mat data_im(width * height * channels, 1, 1);
    data_im.fill(0.0f);

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    data_col = data_col.reshape(data_col_w * data_col_h * data_col_c, 1, 1);

    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col.channel(0)[col_index];
                col2im_add_pixel(data_im.channel(0), height, width, channels, im_row, im_col, c_im, pad, val);
            }
        }
    }
    data_col = data_col.reshape(data_col_w, data_col_h, data_col_c);

    return data_im.reshape(width, height, channels);
}

float SAFE_GET(const ncnn::Mat& input, int x, int y, int c, int H, int W)
{
    if (x >= 0 && x < W && y >= 0 && y < H)
        return input.channel(c)[y * W + x];
    else
        return 0;
}

ncnn::Mat grid_sample(ncnn::Mat& input, ncnn::Mat& grid)
{
    int C = input.c;
    int IH = input.h;
    int IW = input.w;
    int H = grid.c;
    int W = grid.h;
    ncnn::Mat output(input.w, input.h, input.c); output.fill(0.0f);

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float ix = grid.channel(h)[w * 2 + 0];
            float iy = grid.channel(h)[w * 2 + 1];

            ix = ((ix + 1) / 2) * (IW - 1);
            iy = ((iy + 1) / 2) * (IH - 1);

            float ix_nw = std::floor(ix);
            float iy_nw = std::floor(iy);
            float ix_ne = ix_nw + 1;
            float iy_ne = iy_nw;
            float ix_sw = ix_nw;
            float iy_sw = iy_nw + 1;
            float ix_se = ix_nw + 1;
            float iy_se = iy_nw + 1;

            float nw = (ix_se - ix) * (iy_se - iy);
            float ne = (ix - ix_sw) * (iy_sw - iy);
            float sw = (ix_ne - ix) * (iy - iy_ne);
            float se = (ix - ix_nw) * (iy - iy_nw);


            for (int c = 0; c < C; c++) {
                float nw_val = SAFE_GET(input, ix_nw, iy_nw, c, IH, IW);
                float ne_val = SAFE_GET(input, ix_ne, iy_ne, c, IH, IW);
                float sw_val = SAFE_GET(input, ix_sw, iy_sw, c, IH, IW);
                float se_val = SAFE_GET(input, ix_se, iy_se, c, IH, IW);
                output.channel(c)[h * IW + w] = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
            }
        }
    }
    return output;
}

cv::Mat argmaxC(const ncnn::Mat& in)
{
    int C = in.c, H = in.h, W = in.w;
    cv::Mat out = cv::Mat::zeros(H, W, CV_32FC1);
    std::vector<float> temp(C);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++)
                temp[c] = in.channel(c)[h*W+w];
            out.at<float>(h, w) = std::distance(temp.begin(), std::max_element(temp.begin(), temp.end()));
        }
    }
    return out;
}

std::vector<cv::Mat> ncnn2cvC1(ncnn::Mat in)
{
    int C = in.c, H = in.h, W = in.w;
    std::vector<cv::Mat> out(C);
    for (int c = 0; c < C; c++)
        out[c] = cv::Mat(H, W, CV_32FC1, in.channel(c));
    return out;
}

cv::Mat morpho(const cv::Mat& mask)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat tem;
    mask.convertTo(tem, CV_8UC1, 255.0, 0.0);
    cv::dilate(tem, tem, kernel, cv::Point(-1, -1), 5);
    cv::Mat res;
    tem.convertTo(res, CV_32FC1, 1.0 / 255.0, 0.0);
    return res;
}

cv::Mat cv2C3(cv::Mat in)
{
    if (in.type() == CV_32FC1) {
        cv::Mat C3;
        cv::cvtColor(in, C3, cv::COLOR_GRAY2RGB);
        return C3;
    }
    else {
        std::cout << "cv2C3 error, not support!" << std::endl;
        return in;
    }
}

cv::Mat get_average_color(cv::Mat mask, cv::Mat arms)
{
    std::vector<cv::Point> count;
    cv::findNonZero(mask, count);
    if (count.size() < 1) {
        return cv::Mat::zeros(arms.size(), CV_32FC3);
    }
    else {
        cv::Scalar sum4 = cv::sum(arms);
        return cv::Mat(arms.size(), CV_32FC3, cv::Scalar(sum4.val[0] / count.size(), sum4.val[1] / count.size(), sum4.val[2] / count.size()));
    }
}

cv::Mat inference(ncnn::Mat input_semantics, ncnn::Mat ref_semantics, cv::Mat ref_image, cv::Mat real_image, cv::Mat seg_img, cv::Mat ref_seg, cv::Mat img_dp_mask)
{
    // process
    cv::Mat arm_mask = seg_img == 2; arm_mask.convertTo(arm_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat leg_mask = seg_img == 3; leg_mask.convertTo(leg_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat fore_mask = seg_img > 0; fore_mask.convertTo(fore_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat t_mask = ref_seg == 5; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat ref_cl_mask = ref_seg.mul(1 - t_mask) + t_mask * 4;
    ref_cl_mask = ref_cl_mask == 4; ref_cl_mask.convertTo(ref_cl_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat masked_ref = ref_image.mul(cv2C3(ref_cl_mask));
    cv::Mat _masked_ref_ = masked_ref.clone();
    t_mask = ref_seg == 1; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat no_head_mask = ref_seg.mul(1 - t_mask) + t_mask * 0;
    t_mask = ref_seg == 6; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat no_head_and_shoes_mask = no_head_mask.mul(1 - t_mask) + t_mask * 0;
    ncnn::Mat ref_mask = mask_convert(no_head_and_shoes_mask, 7);


    // feature_img_extractor inference
    ncnn::Net feature_img_extractor;
    feature_img_extractor.load_param("assert/feature_img_extractor-sim-opt.param");
    feature_img_extractor.load_model("assert/feature_img_extractor-sim-opt.bin");
    ncnn::Mat adaptive_feature_img;
    {
        ncnn::Extractor ex = feature_img_extractor.create_extractor();
        ex.input("input_img", input_semantics);
        ex.extract("adaptive_feature_img", adaptive_feature_img);
    }


    // feature_ref_extractor inference
    ncnn::Net feature_ref_extractor;
    feature_ref_extractor.load_param("assert/feature_ref_extractor-sim-opt-opt.param");
    feature_ref_extractor.load_model("assert/feature_ref_extractor-sim-opt-opt.bin");
    ncnn::Mat adaptive_feature_ref;
    {
        ncnn::Mat ref_image_mat = cv2ncnn(ref_image);
        ncnn::Extractor ex = feature_ref_extractor.create_extractor();
        ex.input("ref_semantics", ref_semantics);
        ex.input("ref_image", ref_image_mat);
        ex.extract("adaptive_feature_ref", adaptive_feature_ref);
    }


    // process
    ncnn::Mat adf_img_t = im2col_cpu(adaptive_feature_img, 3, 1, 1);
    ncnn::Mat adf_ref = im2col_cpu(adaptive_feature_ref, 3, 1, 1);
    ncnn::Mat c_ref_seg = im2col_cpu(ref_mask, 4, 4, 0);
    ncnn::Mat masked_ref_mat = cv2ncnn(masked_ref); masked_ref_mat = im2col_cpu(masked_ref_mat, 4, 4, 0);
    ncnn::Mat adf_geo_src_f_t = im2col_cpu(adaptive_feature_img, 4, 4, 0);
    ncnn::Mat adf_geo_tar_f = im2col_cpu(adaptive_feature_ref, 4, 4, 0);

    ncnn::Net process1;
    process1.load_param("assert/process1-sim-opt.param");
    process1.load_model("assert/process1-sim-opt.bin");
    ncnn::Mat warp_c, warped_mask, geo_corr;
    {
        ncnn::Extractor ex = process1.create_extractor();
        ex.input("adf_img_t", adf_img_t);
        ex.input("adf_ref", adf_ref);
        ex.input("masked_ref", masked_ref_mat);
        ex.input("c_ref_seg", c_ref_seg);
        ex.input("adf_geo_src_f_t", adf_geo_src_f_t);
        ex.input("adf_geo_tar_f", adf_geo_tar_f);
        ex.extract("warp_c", warp_c);
        ex.extract("warped_mask", warped_mask);
        ex.extract("geo_corr", geo_corr);
    }

    warp_c = col2im_cpu(warp_c, 256, 4, 4, 0);
    warped_mask = col2im_cpu(warped_mask, 256, 4, 4, 0);

    
    // regression inference
    ncnn::Net regression;
    regression.load_param("assert/regression-sim-opt.param");
    regression.load_model("assert/regression-sim-opt.bin");
    ncnn::Mat theta;
    {
        ncnn::Extractor ex = regression.create_extractor();
        ex.input("geo_corr", geo_corr);
        ex.extract("theta", theta);
    }

    
    // gridGen inference
    ncnn::Net gridGen;
    gridGen.load_param("assert/gridGen-sim-opt.param");
    gridGen.load_model("assert/gridGen-sim-opt.bin");
    ncnn::Mat warped_grid;
    {
        ncnn::Extractor ex = gridGen.create_extractor();
        ex.input("theta", theta);
        ex.extract("warped_grid", warped_grid);
    }

    
    // process
    ncnn::Mat ref_cl_mask_mat = cv2ncnn(ref_cl_mask);
    ncnn::Mat TPS_warp_cl_mask = grid_sample(ref_cl_mask_mat, warped_grid);
    ncnn::Mat _masked_ref_mat = cv2ncnn(_masked_ref_);
    ncnn::Mat tps_warp_out = grid_sample(_masked_ref_mat, warped_grid);
    t_mask = seg_img == 5; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat real_cloth_mask = seg_img.mul(1 - t_mask) + t_mask * 4;
    real_cloth_mask = real_cloth_mask == 4; real_cloth_mask.convertTo(real_cloth_mask, CV_32F, 1.0 / 255.0, 0.0);
    t_mask = img_dp_mask == 1; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    no_head_mask = img_dp_mask.mul(1 - t_mask) + t_mask * 0;
    t_mask = img_dp_mask == 6; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat no_head_and_foot_mask = no_head_mask.mul(1 - t_mask) + t_mask * 0;
    ncnn::Mat c_no_head_and_foot_mask = mask_convert(no_head_and_foot_mask, 7);

    
    // netG1 inference
    ncnn::Net netG1;
    netG1.load_param("assert/netG1-sim-opt-opt-opt.param");
    netG1.load_model("assert/netG1-sim-opt-opt-opt.bin");
    ncnn::Mat refine_seg;
    {
        ncnn::Extractor ex = netG1.create_extractor();
        ex.input("warped_mask", warped_mask);
        ex.input("c_no_head_and_foot_mask", c_no_head_and_foot_mask);
        ex.input("input_semantics", input_semantics);
        ex.extract("sigmoid_refine_seg", refine_seg);
    }

    // process
    cv::Mat refine_seg_cv = argmaxC(refine_seg);
    t_mask = refine_seg_cv == 5; t_mask.convertTo(t_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat refine_cloth_mask = refine_seg_cv.mul(1 - t_mask) + t_mask * 4;
    refine_cloth_mask = refine_cloth_mask == 4; refine_cloth_mask.convertTo(refine_cloth_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat refine_arm_mask = refine_seg_cv == 2; refine_arm_mask.convertTo(refine_arm_mask, CV_32F, 1.0 / 255.0, 0.0);
    cv::Mat refine_leg_mask = refine_seg_cv == 3; refine_leg_mask.convertTo(refine_leg_mask, CV_32F, 1.0 / 255.0, 0.0);
    ncnn::Mat masked_label = mask_convert(seg_img.mul(1 - real_cloth_mask), 7);
    std::vector<cv::Mat> masked_label_cv = ncnn2cvC1(masked_label);
    cv::Mat new_arm_mask = refine_arm_mask.mul(1 - masked_label_cv[6]).mul(1 - masked_label_cv[1]);
    cv::Mat new_leg_mask = refine_leg_mask.mul(1 - masked_label_cv[6]).mul(1 - masked_label_cv[1]);
    cv::Mat bigger_real_cloth_mask = morpho(real_cloth_mask);
    cv::Mat image_hole = real_image.mul(cv2C3(1 - bigger_real_cloth_mask)).mul(cv2C3(new_arm_mask));
    image_hole = image_hole + real_image.mul(cv2C3(1 - bigger_real_cloth_mask)).mul(cv2C3(new_leg_mask));
    image_hole = image_hole + real_image.mul(cv2C3(1 - real_cloth_mask)).mul(cv2C3(1 - arm_mask)).mul(cv2C3(1 - leg_mask)).mul(cv2C3(1 - new_arm_mask)).mul(cv2C3(1 - new_leg_mask)).mul(cv2C3(fore_mask));
    masked_label_cv[2] = new_arm_mask.mul(1 - masked_label_cv[1]).mul(1 - masked_label_cv[6]);
    masked_label_cv[3] = new_leg_mask.mul(1 - masked_label_cv[1]).mul(1 - masked_label_cv[6]);
    masked_label_cv[0] = cv::Mat(256, 256, CV_32FC1, 1.0f).mul(1 - masked_label_cv[1]).mul(1 - new_arm_mask).mul(1 - new_leg_mask).mul(1 - masked_label_cv[6]).mul(1 - masked_label_cv[4]).mul(1 - masked_label_cv[5]);
    cv::Mat limb_mask = new_arm_mask + new_leg_mask;
    cv::Mat skin_color = get_average_color((1 - bigger_real_cloth_mask).mul(limb_mask), real_image.mul(cv2C3(1 - bigger_real_cloth_mask)).mul(cv2C3(limb_mask)));
    cv::Mat TPS_refine = cv2C3(ncnn2cv(TPS_warp_cl_mask)).mul(ncnn2cv(tps_warp_out)).mul(cv2C3(real_cloth_mask));
    cv::Mat input_cl = ncnn2cv(warp_c).mul(cv2C3(1 - new_arm_mask)).mul(cv2C3(1 - new_leg_mask)).mul(cv2C3(1 - masked_label_cv[1])).mul(cv2C3(1 - masked_label_cv[6])).mul(cv2C3(refine_cloth_mask));


    // netG2 inference
    ncnn::Net netG2;
    netG2.load_param("assert/netG2-sim-opt-opt-opt-opt.param");
    netG2.load_model("assert/netG2-sim-opt-opt-opt-opt.bin");
    ncnn::Mat initial_fake_image, occlusion_mask;
    {
        int C = masked_label_cv.size(), H = masked_label_cv[0].rows, W = masked_label_cv[0].cols;
        ncnn::Mat masked_label_mat(W, H, C);
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    masked_label_mat.channel(c)[h * W + w] = masked_label_cv[c].at<float>(h, w);
                }
            }
        }
        ncnn::Extractor ex = netG2.create_extractor();
        ex.input("input_cl", cv2ncnn(input_cl));
        ex.input("TPS_refine", cv2ncnn(TPS_refine));
        ex.input("masked_label", masked_label_mat);
        ex.input("image_hole", cv2ncnn(image_hole));
        ex.input("skin_color", cv2ncnn(skin_color));
        ex.input("input_semantics", input_semantics);
        ex.extract("tanh_initial_fake_image", initial_fake_image);
        ex.extract("sigmoid_occlusion_mask", occlusion_mask);
    }


    // process
    cv::Mat occlusion_mask_cv = ncnn2cv(occlusion_mask).mul(refine_cloth_mask);
    cv::Mat fake_result = ncnn2cv(initial_fake_image).mul(cv2C3(1 - occlusion_mask_cv)) + ncnn2cv(tps_warp_out).mul(cv2C3(occlusion_mask_cv));


    // process
    cv::Mat fake = fake_result.clone();
    cv::Mat show;
    fake.convertTo(show, CV_8UC3, 127.5, 127.5);

    
    return show;
}

int main()
{
    ncnn::Mat input_semantics;
    ncnn::Mat ref_semantics;
    cv::Mat ref_image;
    cv::Mat real_image;
    cv::Mat seg_img;
    cv::Mat ref_seg;
    cv::Mat img_dp_mask;

    dataset(input_semantics, ref_semantics, ref_image, real_image, seg_img, ref_seg, img_dp_mask);

    cv::Mat show = inference(input_semantics, ref_semantics, ref_image, real_image, seg_img, ref_seg, img_dp_mask);

    cv::cvtColor(show, show, cv::COLOR_RGB2BGR);
    cv::imwrite("test.png", show);
    

    return 0;
}
