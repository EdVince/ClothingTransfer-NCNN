#pragma once

#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

class CTNet
{
public:
	CTNet();
	~CTNet();

	cv::Mat call(cv::Mat src_rgb, cv::Mat dst_rgb, cv::Mat src_candidate, cv::Mat src_subset, cv::Mat dst_candidate, cv::Mat dst_subset, cv::Mat src_seg, cv::Mat dst_seg, cv::Mat src_iuv);

private:
	ncnn::Mat get_label_tensor(cv::Mat candidate, cv::Mat subset, cv::Mat img);
	cv::Mat convert_seg(const cv::Mat& t_seg);
	cv::Mat convert_dp_mask(const cv::Mat& dp);
	ncnn::Mat mask_convert(const cv::Mat& seg, int c);
	float im2col_get_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad);
	ncnn::Mat im2col_cpu(ncnn::Mat& data_im, int ksize, int stride, int pad);
	ncnn::Mat cv2ncnn(const cv::Mat& in);
	cv::Mat ncnn2cv(const ncnn::Mat& in);
	void col2im_add_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad, float val);
	ncnn::Mat col2im_cpu(ncnn::Mat& data_col, int output_size, int ksize, int stride, int pad);
	float SAFE_GET(const ncnn::Mat& input, int x, int y, int c, int H, int W);
	ncnn::Mat grid_sample(ncnn::Mat& input, ncnn::Mat& grid);
	cv::Mat argmaxC(const ncnn::Mat& in);
	std::vector<cv::Mat> ncnn2cvC1(ncnn::Mat in);
	cv::Mat morpho(const cv::Mat& mask);
	cv::Mat cv2C3(cv::Mat in);
	cv::Mat get_average_color(cv::Mat mask, cv::Mat arms);

	void dataset(ncnn::Mat& input_semantics, ncnn::Mat& ref_semantics, cv::Mat& ref_image, cv::Mat& real_image, cv::Mat& seg_img, cv::Mat& ref_seg, cv::Mat& img_dp_mask,
		cv::Mat src_rgb, cv::Mat dst_rgb, cv::Mat src_candidate, cv::Mat src_subset, cv::Mat dst_candidate, cv::Mat dst_subset, cv::Mat src_seg, cv::Mat dst_seg, cv::Mat src_iuv);
	cv::Mat inference(ncnn::Mat input_semantics, ncnn::Mat ref_semantics, cv::Mat ref_image, cv::Mat real_image, cv::Mat seg_img, cv::Mat ref_seg, cv::Mat img_dp_mask);

private:

	ncnn::Net feature_img_extractor;
	ncnn::Net feature_ref_extractor;
	ncnn::Net process1;
	ncnn::Net regression;
	ncnn::Net gridGen;
	ncnn::Net netG1;
	ncnn::Net netG2;


};

