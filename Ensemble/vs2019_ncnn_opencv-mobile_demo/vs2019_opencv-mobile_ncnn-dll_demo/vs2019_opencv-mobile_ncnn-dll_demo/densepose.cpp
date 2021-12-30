#include "densepose.h"

DensePose::DensePose()
{
    backbone.load_param("assert/DensePose/backbone-sim-opt.param");
    backbone.load_model("assert/DensePose/backbone-sim-opt.bin");

    box_head.load_param("assert/DensePose/box_head-sim-opt-opt.param");
    box_head.load_model("assert/DensePose/box_head-sim-opt-opt.bin");

    box_predictor.load_param("assert/DensePose/box_predictor-sim-opt.param");
    box_predictor.load_model("assert/DensePose/box_predictor-sim-opt.bin");

    decoder.load_param("assert/DensePose/decoder-sim-opt.param");
    decoder.load_model("assert/DensePose/decoder-sim-opt.bin");

    densepose_head.load_param("assert/DensePose/densepose_head-sim-opt.param");
    densepose_head.load_model("assert/DensePose/densepose_head-sim-opt.bin");

    densepose_predictor.load_param("assert/DensePose/densepose_predictor-sim-opt.param");
    densepose_predictor.load_model("assert/DensePose/densepose_predictor-sim-opt.bin");

    rpn_head.load_param("assert/DensePose/rpn_head-sim-opt.param");
    rpn_head.load_model("assert/DensePose/rpn_head-sim-opt.bin");

    // anchor_generator
    int anchors_size[5] = { 120000,30000,7500,1875,507 };
    anchors.resize(5);
    for (int i = 0; i < 5; i++) {
        anchors[i] = ncnn::Mat(4, anchors_size[i], 1);
        std::ifstream in("assert/DensePose/anchors_" + std::to_string(i) + ".data", std::ios::in | std::ios::binary);
        in.read((char*)anchors[i].channel(0), anchors_size[i] * 4 * 4);
    }
}
DensePose::~DensePose()
{
    backbone.clear();
    box_head.clear();
    box_predictor.clear();
    decoder.clear();
    densepose_head.clear();
    densepose_predictor.clear();
    rpn_head.clear();
}

cv::Mat DensePose::apply_deltas(const cv::Mat& deltas, const cv::Mat& boxes, float wx, float wy, float ww, float wh, float scale_clamp)
{
    cv::Mat widths = boxes(cv::Range::all(), cv::Range(2, 3)).clone() - boxes(cv::Range::all(), cv::Range(0, 1)).clone();
    cv::Mat heights = boxes(cv::Range::all(), cv::Range(3, 4)).clone() - boxes(cv::Range::all(), cv::Range(1, 2)).clone();
    cv::Mat ctr_x = boxes(cv::Range::all(), cv::Range(0, 1)).clone() + 0.5 * widths;
    cv::Mat ctr_y = boxes(cv::Range::all(), cv::Range(1, 2)).clone() + 0.5 * heights;

    cv::Mat dx = deltas(cv::Range::all(), cv::Range(0, 1)).clone() / wx;
    cv::Mat dy = deltas(cv::Range::all(), cv::Range(1, 2)).clone() / wy;
    cv::Mat dw = deltas(cv::Range::all(), cv::Range(2, 3)).clone() / ww;
    cv::Mat dh = deltas(cv::Range::all(), cv::Range(3, 4)).clone() / wh;

    cv::threshold(dw, dw, scale_clamp, scale_clamp, cv::THRESH_TRUNC);
    cv::threshold(dh, dh, scale_clamp, scale_clamp, cv::THRESH_TRUNC);

    cv::exp(dw, dw);
    cv::exp(dh, dh);
    cv::Mat pred_ctr_x = dx.mul(widths) + ctr_x;
    cv::Mat pred_ctr_y = dy.mul(heights) + ctr_y;
    cv::Mat pred_w = dw.mul(widths);
    cv::Mat pred_h = dh.mul(heights);

    cv::Mat x1 = pred_ctr_x - 0.5 * pred_w;
    cv::Mat y1 = pred_ctr_y - 0.5 * pred_h;
    cv::Mat x2 = pred_ctr_x + 0.5 * pred_w;
    cv::Mat y2 = pred_ctr_y + 0.5 * pred_h;

    cv::Mat result;
    cv::hconcat(std::vector<cv::Mat>{x1, y1, x2, y2}, result);

    return result;
}

float DensePose::IOU(std::vector<float>& A, std::vector<float>& B)
{
    // 左上右下坐标(x1,y1,x2,y2)
    float w = std::max(0.0f, std::min(A[2], B[2]) - std::max(A[0], B[0]) + 1);
    float h = std::max(0.0f, std::min(A[3], B[3]) - std::max(A[1], B[1]) + 1);
    float area1 = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
    float area2 = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
    float inter = w * h;
    float iou = inter / (area1 + area2 - inter);
    return iou;
}

bool DensePose::sort_score(std::vector<float>& box1, std::vector<float>& box2)
{
    return (box1[4] > box2[4]);
}

void DensePose::nms(std::vector<std::vector<float> >& vec_boxs, float thresh)
{
    // box[5]: x1, y1, x2, y2, score
    // 按分值从大到小排序
    //std::sort(vec_boxs.begin(), vec_boxs.end(), sort_score);
    std::sort(vec_boxs.begin(), vec_boxs.end(), [](std::vector<float>& box1, std::vector<float>& box2) {return (box1[4] > box2[4]); });
    //标志，false代表留下，true代表扔掉
    std::vector<bool> del(vec_boxs.size(), false);
    for (size_t i = 0; i < vec_boxs.size() - 1; i++) {
        if (!del[i]) {
            for (size_t j = i + 1; j < vec_boxs.size(); j++) {
                if (!del[j] && IOU(vec_boxs[i], vec_boxs[j]) >= thresh) {
                    del[j] = true; //IOU大于阈值扔掉
                }
            }
        }
    }
    std::vector<std::vector<float>> result;
    for (size_t i = 0; i < vec_boxs.size(); i++) {
        if (!del[i]) {
            result.push_back(vec_boxs[i]);
        }
    }
    vec_boxs.clear();
    std::vector<std::vector<float> >().swap(vec_boxs);
    vec_boxs = result;
}

std::vector<std::vector<float>> DensePose::find_top_rpn_proposals(const std::vector<cv::Mat>& proposals, const std::vector<cv::Mat>& pred_objectness_logits, int image_sizes, float nms_thresh, int pre_nms_topk, int post_nms_topk, float min_box_size)
{
    // 1. Select top-k anchor for every level and every image
    std::vector<cv::Mat> topk_proposals_vector;
    std::vector<cv::Mat> topk_scores_vector;
    std::vector<cv::Mat> level_ids_vector;
    for (int level_id = 0; level_id < proposals.size(); level_id++) {
        auto proposals_i = proposals[level_id];
        auto logits_i = pred_objectness_logits[level_id];

        int num_proposals_i = std::min(pre_nms_topk, logits_i.cols);

        cv::Mat topk_scores_i, topk_idx;
        cv::sort(logits_i, topk_scores_i, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
        cv::sortIdx(logits_i, topk_idx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
        topk_scores_i = topk_scores_i(cv::Range::all(), cv::Range(0, num_proposals_i));
        topk_idx = topk_idx(cv::Range::all(), cv::Range(0, num_proposals_i));

        cv::Mat topk_proposals_i;
        for (int i = 0; i < num_proposals_i; i++)
            topk_proposals_i.push_back(proposals_i.row(((int*)topk_idx.data)[i]));

        topk_proposals_vector.push_back(topk_proposals_i);
        topk_scores_vector.push_back(topk_scores_i);
        level_ids_vector.push_back(cv::Mat(num_proposals_i, 1, CV_32SC1, cv::Scalar(level_id)));
    }

    // 2. Concat all levels together
    cv::Mat topk_scores, topk_proposals, level_ids;
    cv::hconcat(topk_scores_vector, topk_scores);
    cv::vconcat(topk_proposals_vector, topk_proposals);
    cv::vconcat(level_ids_vector, level_ids);

    // 3. For each image, run a per-level NMS, and choose topk results.
    auto boxes = topk_proposals;
    auto scores_per_img = topk_scores;
    auto lvl = level_ids;

    boxes = cv::min(cv::max(boxes, 0), 800);

    std::vector<std::vector<float>> bbox;
    for (int i = 0; i < boxes.rows; i++)
    {
        cv::Mat xyxy = boxes.row(i);
        float x1 = xyxy.at<float>(0, 0), y1 = xyxy.at<float>(0, 1), x2 = xyxy.at<float>(0, 2), y2 = xyxy.at<float>(0, 3);
        float widths = x2 - x1, heights = y2 - y1;
        if (widths > 0.0 && heights > 0.0)
            bbox.push_back(std::vector<float>{x1, y1, x2, y2, scores_per_img.at<float>(0, i)});
    }

    nms(bbox, nms_thresh);
    if (post_nms_topk < bbox.size())
        bbox.resize(post_nms_topk);

    return bbox;
}

cv::Mat DensePose::convert_boxes_to_pooler_format(const cv::Mat& box_lists)
{
    auto boxes = box_lists;
    int sizes = boxes.rows;
    cv::Mat indices = cv::Mat::zeros(sizes, 1, CV_32FC1);
    cv::hconcat(indices, boxes, indices);
    return indices;
}

cv::Mat DensePose::assign_boxes_to_levels(const cv::Mat& box_lists, int min_level, int max_level, int canonical_box_size, int canonical_level)
{
    // area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    cv::Mat x1 = box_lists(cv::Range::all(), cv::Range(0, 1));
    cv::Mat y1 = box_lists(cv::Range::all(), cv::Range(1, 2));
    cv::Mat x2 = box_lists(cv::Range::all(), cv::Range(2, 3));
    cv::Mat y2 = box_lists(cv::Range::all(), cv::Range(3, 4));
    cv::Mat area = (x2 - x1).mul(y2 - y1);
    cv::Mat box_sizes;
    cv::sqrt(area, box_sizes);

    cv::Mat level_assignments;
    cv::log(box_sizes / canonical_box_size + 1e-8, level_assignments);
    level_assignments = canonical_level + level_assignments / std::log(2);
    for (int i = 0; i < level_assignments.rows; i++)
        for (int j = 0; j < level_assignments.cols; j++)
            level_assignments.at<float>(i, j) = std::floor(level_assignments.at<float>(i, j));

    level_assignments = cv::max(cv::min(level_assignments, max_level), min_level);

    level_assignments.convertTo(level_assignments, CV_32SC1);
    level_assignments = level_assignments - min_level;

    return level_assignments;
}

void DensePose::pre_calc_for_bilinear_interpolate(const int height, const int width, const int pooled_height, const int pooled_width, const int iy_upper, const int ix_upper, float roi_start_h, float roi_start_w, float bin_size_h, float bin_size_w, int roi_bin_grid_h, int roi_bin_grid_w, std::vector<PreCalc>& pre_calc) {
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
            for (int iy = 0; iy < iy_upper; iy++) {
                const float yy = roi_start_h + ph * bin_size_h +
                    static_cast<float>(iy + .5f) * bin_size_h /
                    static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++) {
                    const float xx = roi_start_w + pw * bin_size_w +
                        static_cast<float>(ix + .5f) * bin_size_w /
                        static_cast<float>(roi_bin_grid_w);

                    float x = xx;
                    float y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        // empty
                        PreCalc pc;
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0;
                        pc.w2 = 0;
                        pc.w3 = 0;
                        pc.w4 = 0;
                        pre_calc[pre_calc_index] = pc;
                        pre_calc_index += 1;
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    int y_low = (int)y;
                    int x_low = (int)x;
                    int y_high;
                    int x_high;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y = (float)y_low;
                    }
                    else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x = (float)x_low;
                    }
                    else {
                        x_high = x_low + 1;
                    }

                    float ly = y - y_low;
                    float lx = x - x_low;
                    float hy = 1. - ly, hx = 1. - lx;
                    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                    // save weights and indeces
                    PreCalc pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = w1;
                    pc.w2 = w2;
                    pc.w3 = w3;
                    pc.w4 = w4;
                    pre_calc[pre_calc_index] = pc;

                    pre_calc_index += 1;
                }
            }
        }
    }
}

void DensePose::ROIAlignForward_cpu_kernel(const int nthreads, const float* bottom_data, const float& spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, const float* bottom_rois, float* top_data)
{
    int roi_cols = 5;

    int n_rois = nthreads / channels / pooled_width / pooled_height;

    for (int n = 0; n < n_rois; n++) {
        int index_n = n * channels * pooled_width * pooled_height;

        const float* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = 0;
        if (roi_cols == 5) {
            roi_batch_ind = offset_bottom_rois[0];
            offset_bottom_rois++;
        }

        // Do not using rounding; this implementation detail is critical
        float roi_start_w = offset_bottom_rois[0] * spatial_scale;
        float roi_start_h = offset_bottom_rois[1] * spatial_scale;
        float roi_end_w = offset_bottom_rois[2] * spatial_scale;
        float roi_end_h = offset_bottom_rois[3] * spatial_scale;

        // Force malformed ROIs to be 1x1
        float roi_width = std::max(roi_end_w - roi_start_w, (float)1.);
        float roi_height = std::max(roi_end_h - roi_start_h, (float)1.);
        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        // we want to precalculate indeces and weights shared by all chanels,
        // this is the key point of optimiation
        std::vector<PreCalc> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc);

        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * pooled_width * pooled_height;
            const float* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
                for (int pw = 0; pw < pooled_width; pw++) {
                    int index = index_n_c + ph * pooled_width + pw;

                    float output_val = 0.;
                    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                            PreCalc pc = pre_calc[pre_calc_index];
                            output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                                pc.w2 * offset_bottom_data[pc.pos2] +
                                pc.w3 * offset_bottom_data[pc.pos3] +
                                pc.w4 * offset_bottom_data[pc.pos4];

                            pre_calc_index += 1;
                        }
                    }
                    output_val /= count;

                    top_data[index] = output_val;
                } // for pw
            } // for ph
        } // for c
    } // for n
}

void DensePose::ROIAlign_forward_cpu(const ncnn::Mat& input, const cv::Mat& rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, float* output)
{

    int num_rois = 1;
    int channels = input.c;
    int height = input.h;
    int width = input.w;

    int output_size = num_rois * pooled_height * pooled_width * channels;

    ROIAlignForward_cpu_kernel(
        output_size,
        input.reshape(input.w * input.h * input.c, 1, 1).channel(0),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        (float*)rois.data,
        output);
}

ncnn::Mat DensePose::my_box_pooler(const std::vector<ncnn::Mat>& x, const cv::Mat& box_lists)
{
    auto pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists);
    auto level_assignments = assign_boxes_to_levels(box_lists, 2, 5, 224, 4);
    ncnn::Mat output(7 * 7 * 256, 1, 1000);
    for (int level = 0; level < 4; level++) {
        std::vector<cv::Point> inds;
        cv::findNonZero(level_assignments == level, inds);
        for (int i = 0; i < inds.size(); i++) {
            ROIAlign_forward_cpu(x[level], pooler_fmt_boxes(cv::Range(inds[i].y, inds[i].y + 1), cv::Range::all()), 0.25 / std::pow(2, level), 7, 7, 2, output.channel(inds[i].y));
        }
    }
    return output;
}

cv::Mat DensePose::predict_boxes(ncnn::Mat& proposal_deltas, const std::vector<std::vector<float>>& proposals)
{
    cv::Mat cv_proposal_deltas(1000, 4, CV_32FC1, proposal_deltas.channel(0));
    cv::Mat proposal_boxes(0, 4, CV_32FC1);
    for (int i = 0; i < proposals.size(); i++) {
        cv::Mat row = (cv::Mat_<float>(1, 4) << proposals[i][0], proposals[i][1], proposals[i][2], proposals[i][3]);
        proposal_boxes.push_back(row);
    }
    cv::Mat predict_boxes = apply_deltas(cv_proposal_deltas, proposal_boxes, 10.0, 10.0, 5.0, 5.0, 4.135166556742356);
    return predict_boxes;
}

std::vector<std::vector<float>> DensePose::fast_rcnn_inference_single_image(cv::Mat boxes, cv::Mat scores, int image_shape, float score_thresh, float nms_thresh, int topk_per_image)
{
    scores = scores(cv::Range::all(), cv::Range(0, scores.cols - 1)).clone();
    boxes = cv::max(cv::min(boxes, image_shape), 0);

    cv::Mat filter_mask = scores > score_thresh;
    std::vector<cv::Point> filter_inds;
    cv::findNonZero(filter_mask, filter_inds);

    cv::Mat temp_boxes(0, 4, CV_32FC1);
    cv::Mat temp_scores(0, 1, CV_32FC1);
    for (auto& x : filter_inds) {
        temp_boxes.push_back(boxes.row(x.y));
        temp_scores.push_back(scores.row(x.y));
    }
    boxes = temp_boxes.clone();
    scores = temp_scores.clone();

    std::vector<std::vector<float>> bbox;
    for (int i = 0; i < boxes.rows; i++)
    {
        cv::Mat xyxy = boxes.row(i);
        float x1 = xyxy.at<float>(0, 0), y1 = xyxy.at<float>(0, 1), x2 = xyxy.at<float>(0, 2), y2 = xyxy.at<float>(0, 3);
        bbox.push_back(std::vector<float>{x1, y1, x2, y2, scores.at<float>(i, 0)});
    }
    nms(bbox, nms_thresh);
    if (topk_per_image < bbox.size())
        bbox.resize(topk_per_image);

    return bbox;
}

std::vector<std::vector<float>> DensePose::my_box_predictor(ncnn::Mat& scores, ncnn::Mat& proposal_deltas, const std::vector<std::vector<float>>& proposals)
{
    cv::Mat cv_boxes = predict_boxes(proposal_deltas, proposals);
    // predict_probs的softmax融进去ncnn里面做了，这里简单的转成cv::Mat给后面用
    cv::Mat cv_scores(1000, 2, CV_32FC1, scores.channel(0));

    auto pred_instances = fast_rcnn_inference_single_image(cv_boxes, cv_scores, 800, 0.05, 0.5, 100);

    return pred_instances;
}

std::vector<std::vector<float>> DensePose::_forward_box(const std::vector<ncnn::Mat>& features, std::vector<std::vector<float>>& proposals)
{
    std::vector<ncnn::Mat> _features_ = { features[0],features[1],features[2],features[3] };
    cv::Mat _proposal_boxes_(0, 4, CV_32FC1);
    for (unsigned int i = 0; i < proposals.size(); ++i)
        _proposal_boxes_.push_back(cv::Mat(1, 4, CV_32FC1, proposals[i].data()));

    ncnn::Mat box_features = my_box_pooler(_features_, _proposal_boxes_);

    box_features = box_features.reshape(12544, 1000);
    ncnn::Mat out_box_features;
    {
        ncnn::Extractor ex = box_head.create_extractor();
        ex.input("in_box_features", box_features);
        ex.extract("out_box_features", out_box_features);
    }

    ncnn::Mat proposal_deltas, scores;
    {
        ncnn::Extractor ex = box_predictor.create_extractor();
        ex.input("box_features", out_box_features);
        ex.extract("proposal_deltas", proposal_deltas);
        ex.extract("scores", scores);
    }

    std::vector<std::vector<float>> pred_instances = my_box_predictor(scores, proposal_deltas, proposals);

    return pred_instances;
}

ncnn::Mat DensePose::my_densepose_pooler(const ncnn::Mat& x, const cv::Mat& box_lists)
{
    cv::Mat pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists);

    int num = pooler_fmt_boxes.rows;

    // 这里有问题，为了保证后面的代码运行征程，这里num一定要是1，也就是只能有一个人，不然后面会crush
    if (num != 1) {
        std::cout << "error" << std::endl;
        return ncnn::Mat(0, 0, 0);
    }

    ncnn::Mat output(28 * 28 * 256, 1, num);
    for (int n = 0; n < num; n++) {
        ROIAlign_forward_cpu(x, pooler_fmt_boxes(cv::Range(n, n + 1), cv::Range::all()), 0.25, 28, 28, 2, output.channel(n));
    }
    output = output.reshape(28, 28, 256);

    return output;
}

std::vector<ncnn::Mat> DensePose::_forward_densepose(const std::vector<ncnn::Mat>& features, std::vector<std::vector<float>>& instances)
{
    ncnn::Mat features_list;
    {
        ncnn::Extractor ex = decoder.create_extractor();
        ex.input("p2", features[0]);
        ex.input("p3", features[1]);
        ex.input("p4", features[2]);
        ex.input("p5", features[3]);
        ex.extract("features_list", features_list);
    }

    cv::Mat pred_boxes(0, 4, CV_32FC1);
    for (unsigned int i = 0; i < instances.size(); ++i)
        pred_boxes.push_back(cv::Mat(1, 4, CV_32FC1, instances[i].data()));

    ncnn::Mat features_dp = my_densepose_pooler(features_list, pred_boxes);

    ncnn::Mat densepose_head_outputs;
    {
        ncnn::Extractor ex = densepose_head.create_extractor();
        ex.input("features_dp", features_dp);
        ex.extract("densepose_head_outputs", densepose_head_outputs);
    }

    std::vector<ncnn::Mat> densepose_predictor_outputs(4);
    {
        ncnn::Extractor ex = densepose_predictor.create_extractor();
        ex.input("densepose_head_outputs", densepose_head_outputs);
        ex.extract("coarse_segm", densepose_predictor_outputs[0]);
        ex.extract("fine_segm", densepose_predictor_outputs[1]);
        ex.extract("u", densepose_predictor_outputs[2]);
        ex.extract("v", densepose_predictor_outputs[3]);
    }

    if (instances.size() != 1) {
        std::cout << "error" << std::endl;
    }

    // 数据拷贝，不然会错误
    std::vector<ncnn::Mat> result;
    ncnn::Mat coarse_segm(densepose_predictor_outputs[0].w, densepose_predictor_outputs[0].h, densepose_predictor_outputs[0].c);
    for (int c = 0; c < coarse_segm.c; c++)
        for (int i = 0; i < coarse_segm.h * coarse_segm.w; i++)
            coarse_segm.channel(c)[i] = densepose_predictor_outputs[0].channel(c)[i];
    result.push_back(coarse_segm);

    ncnn::Mat fine_segm(densepose_predictor_outputs[1].w, densepose_predictor_outputs[1].h, densepose_predictor_outputs[1].c);
    for (int c = 0; c < fine_segm.c; c++)
        for (int i = 0; i < fine_segm.h * fine_segm.w; i++)
            fine_segm.channel(c)[i] = densepose_predictor_outputs[1].channel(c)[i];
    result.push_back(fine_segm);

    ncnn::Mat u(densepose_predictor_outputs[2].w, densepose_predictor_outputs[2].h, densepose_predictor_outputs[2].c);
    for (int c = 0; c < u.c; c++)
        for (int i = 0; i < u.h * u.w; i++)
            u.channel(c)[i] = densepose_predictor_outputs[2].channel(c)[i];
    result.push_back(u);

    ncnn::Mat v(densepose_predictor_outputs[3].w, densepose_predictor_outputs[3].h, densepose_predictor_outputs[3].c);
    for (int c = 0; c < v.c; c++)
        for (int i = 0; i < v.h * v.w; i++)
            v.channel(c)[i] = densepose_predictor_outputs[3].channel(c)[i];
    result.push_back(v);

    ncnn::Mat instance(5, 1, 1);
    instance.channel(0)[0] = instances[0][0];
    instance.channel(0)[1] = instances[0][1];
    instance.channel(0)[2] = instances[0][2];
    instance.channel(0)[3] = instances[0][3];
    instance.channel(0)[4] = instances[0][4];
    result.push_back(instance);

    return result;
}

void DensePose::_postprocess(std::vector<ncnn::Mat>& instances)
{
    float scale = 256.0f / 800.0f;
    for (int i = 0; i < 4; i++) {
        instances[4].channel(0)[i] *= scale;
        instances[4].channel(0)[i] = std::max(0.0f, std::min(instances[4].channel(0)[i], 256.0f));
    }
}

cv::Mat DensePose::argmax(const ncnn::Mat& in)
{
    int C = in.c, H = in.h, W = in.w;
    cv::Mat idx(H, W, CV_8UC1);
    std::vector<float> temp(C);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                temp[c] = in.channel(c)[h * W + w];
            }
            idx.at<uchar>(h, w) = std::distance(temp.begin(), std::max_element(temp.begin(), temp.end()));
        }
    }
    return idx;
}

cv::Mat DensePose::generate(const std::vector<ncnn::Mat> SIUVxyxys)
{
    ncnn::Mat S = SIUVxyxys[0];
    ncnn::Mat I = SIUVxyxys[1];
    ncnn::Mat U = SIUVxyxys[2];
    ncnn::Mat V = SIUVxyxys[3];

    cv::Mat personS = argmax(S);
    cv::Mat personI = argmax(I);

    auto mask = personS;
    personI = personI.mul(mask);

    cv::Mat personU = cv::Mat::zeros(personS.size(), CV_32FC1);
    cv::Mat personV = cv::Mat::zeros(personS.size(), CV_32FC1);

    for (int partId = 0; partId < 25; partId++) {
        cv::Mat personIpartId = personI == partId;
        std::vector<cv::Point> idx;
        cv::findNonZero(personIpartId, idx);
        for (auto& l : idx) {
            personU.at<float>(l) = std::min(1.0f, std::max(0.0f, U.channel(partId)[l.y * 112 + l.x]));
            personV.at<float>(l) = std::min(1.0f, std::max(0.0f, V.channel(partId)[l.y * 112 + l.x]));
        }
    }

    mask.convertTo(mask, CV_32FC1);

    personU = personU.mul(mask);
    personV = personV.mul(mask);

    personU.convertTo(personU, CV_8UC1, 255.0);
    personV.convertTo(personV, CV_8UC1, 255.0);

    cv::Mat iuv;
    std::vector<cv::Mat> IUV = { personI, personU, personV };
    cv::merge(IUV, iuv);

    int x1 = std::round(SIUVxyxys[4].channel(0)[0]);
    int y1 = std::round(SIUVxyxys[4].channel(0)[1]);
    int x2 = std::round(SIUVxyxys[4].channel(0)[2]);
    int y2 = std::round(SIUVxyxys[4].channel(0)[3]);

    cv::resize(iuv, iuv, cv::Size(x2 - x1, y2 - y1));

    cv::Mat showImage = cv::Mat::zeros(256, 256, CV_8UC3);
    iuv.copyTo(showImage(cv::Range(y1, y2), cv::Range(x1, x2)));

    return showImage;
}

cv::Mat DensePose::call(cv::Mat rgb)
{
    cv::Mat image;
    cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);


    // ************************************************ preprocess_image
    const float mean_vals[3] = { 103.5300f, 116.2800f, 123.6750f };
    const float norm_vals[3] = { 1.f, 1.f, 1.f };
    ncnn::Mat images = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PixelType::PIXEL_BGR, 256, 256, 800, 800);
    images.substract_mean_normalize(mean_vals, norm_vals);


    // ************************************************ backbone
    std::vector<ncnn::Mat> features(5);
    {
        ncnn::Extractor ex = backbone.create_extractor();
        ex.input("images", images);
        for (int i = 0; i < 5; i++)
            ex.extract(("features_p" + std::to_string(i + 2)).c_str(), features[i]);
    }


    // ************************************************ proposal_generator

    // rpn_head
    std::vector<ncnn::Mat> pred_objectness_logits(5);
    std::vector<ncnn::Mat> pred_anchor_deltas(5);
    {
        ncnn::Extractor ex = rpn_head.create_extractor();
        for (int i = 0; i < 5; i++)
            ex.input(("p" + std::to_string(i + 2)).c_str(), features[i]);
        for (int i = 0; i < 5; i++) {
            ex.extract(("pol_" + std::to_string(i)).c_str(), pred_objectness_logits[i]);
            ex.extract(("pad_" + std::to_string(i)).c_str(), pred_anchor_deltas[i]);
        }
    }

    // decode_proposals
    std::vector<cv::Mat> pred_proposals(5);
    for (int i = 0; i < 5; i++) {
        cv::Mat pred_anchor_deltas_i(cv::Size(pred_anchor_deltas[i].w, pred_anchor_deltas[i].h), CV_32FC1, pred_anchor_deltas[i].channel(0));
        cv::Mat anchors_i(cv::Size(anchors[i].w, anchors[i].h), CV_32FC1, anchors[i].channel(0));
        pred_proposals[i] = apply_deltas(pred_anchor_deltas_i, anchors_i, 1.0, 1.0, 1.0, 1.0, 4.135166556742356);
    }

    // find_top_rpn_proposals
    std::vector<cv::Mat> pred_objectness_logits_cvMat(5);
    for (int i = 0; i < 5; i++)
        pred_objectness_logits_cvMat[i] = cv::Mat(cv::Size(pred_objectness_logits[i].w, pred_objectness_logits[i].h), CV_32FC1, pred_objectness_logits[i].channel(0));
    std::vector<std::vector<float>> proposals = find_top_rpn_proposals(pred_proposals, pred_objectness_logits_cvMat, 800, 0.7, 1000, 1000, 0.0);


    // ************************************************ roi_heads
    // _forward_box, _forward_densepose
    std::vector<std::vector<float>> pred_instances = _forward_box(features, proposals);
    std::vector<ncnn::Mat> result = _forward_densepose(features, pred_instances);


    // ************************************************ postprocess
    _postprocess(result);


    // ************************************************ generate iuv image
    cv::Mat iuv = generate(result);


    return iuv;
}