#pragma once

#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>

class LIPJPPNet
{
public:
	LIPJPPNet();
	~LIPJPPNet();

	cv::Mat call(cv::Mat rgb);

private:
	ncnn::Mat inference(cv::Mat image, int w, int h, std::string node);
	ncnn::Mat flip(const ncnn::Mat& mask);

private:
	const float mean_vals[3] = { 104.00698793f, 116.66876762f, 122.67891434f };
	const float norm_vals[3] = { 1.f, 1.f, 1.f };
	const int label_colours[20][3] = { {0, 0, 0}, {128, 0, 0}, {255, 0, 0}, {0, 85, 0}, {170, 0, 51}, {255, 85, 0}, {0, 0, 85}, {0, 119, 221}, {85, 85, 0}, {0, 85, 85}, {85, 51, 0}, {52, 86, 128}, {0, 128, 0}, {0, 0, 255}, {51, 170, 221}, {0, 255, 255}, {85, 255, 170}, {170, 255, 85}, {255, 255, 0}, {255, 170, 0} };

	ncnn::Net net;
};

