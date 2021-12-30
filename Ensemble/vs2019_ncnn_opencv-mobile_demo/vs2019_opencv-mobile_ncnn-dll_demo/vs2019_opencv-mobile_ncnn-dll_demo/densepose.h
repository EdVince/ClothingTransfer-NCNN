#pragma once

#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

class DensePose
{
public:
	DensePose();
	~DensePose();

	cv::Mat call(cv::Mat rgb);

private:
	cv::Mat apply_deltas(const cv::Mat& deltas, const cv::Mat& boxes, float wx, float wy, float ww, float wh, float scale_clamp);
	float IOU(std::vector<float>& A, std::vector<float>& B);
	bool sort_score(std::vector<float>& box1, std::vector<float>& box2);
	void nms(std::vector<std::vector<float> >& vec_boxs, float thresh);
	std::vector<std::vector<float>> find_top_rpn_proposals(const std::vector<cv::Mat>& proposals, const std::vector<cv::Mat>& pred_objectness_logits, int image_sizes, float nms_thresh, int pre_nms_topk, int post_nms_topk, float min_box_size);
	cv::Mat convert_boxes_to_pooler_format(const cv::Mat& box_lists);
	cv::Mat assign_boxes_to_levels(const cv::Mat& box_lists, int min_level, int max_level, int canonical_box_size, int canonical_level);

	struct PreCalc {
		int pos1;
		int pos2;
		int pos3;
		int pos4;
		float w1;
		float w2;
		float w3;
		float w4;
	};
	void pre_calc_for_bilinear_interpolate(const int height, const int width, const int pooled_height, const int pooled_width, const int iy_upper, const int ix_upper, float roi_start_h, float roi_start_w, float bin_size_h, float bin_size_w, int roi_bin_grid_h, int roi_bin_grid_w, std::vector<PreCalc>& pre_calc);
	void ROIAlignForward_cpu_kernel(const int nthreads, const float* bottom_data, const float& spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, const float* bottom_rois, float* top_data);
	void ROIAlign_forward_cpu(const ncnn::Mat& input, const cv::Mat& rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, float* output);
	ncnn::Mat my_box_pooler(const std::vector<ncnn::Mat>& x, const cv::Mat& box_lists);
	cv::Mat predict_boxes(ncnn::Mat& proposal_deltas, const std::vector<std::vector<float>>& proposals);
	std::vector<std::vector<float>> fast_rcnn_inference_single_image(cv::Mat boxes, cv::Mat scores, int image_shape, float score_thresh, float nms_thresh, int topk_per_image);
	std::vector<std::vector<float>> my_box_predictor(ncnn::Mat& scores, ncnn::Mat& proposal_deltas, const std::vector<std::vector<float>>& proposals);
	std::vector<std::vector<float>> _forward_box(const std::vector<ncnn::Mat>& features, std::vector<std::vector<float>>& proposals);
	ncnn::Mat my_densepose_pooler(const ncnn::Mat& x, const cv::Mat& box_lists);
	std::vector<ncnn::Mat> _forward_densepose(const std::vector<ncnn::Mat>& features, std::vector<std::vector<float>>& instances);
	void _postprocess(std::vector<ncnn::Mat>& instances);
	cv::Mat argmax(const ncnn::Mat& in);
	cv::Mat generate(const std::vector<ncnn::Mat> SIUVxyxys);

private:

	ncnn::Net backbone;
	ncnn::Net box_head;
	ncnn::Net box_predictor;
	ncnn::Net decoder;
	ncnn::Net densepose_head;
	ncnn::Net densepose_predictor;
	ncnn::Net rpn_head;

	std::vector<ncnn::Mat> anchors;

};

