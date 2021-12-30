#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>

#include "openpose.h"
#include "lipjppt.h"
#include "densepose.h"
#include "ctnet.h"

int main()
{
    OpenPose openpose;
    LIPJPPNet lipjpp;
    DensePose densepose;
    CTNet ct;


    // read image
    cv::Mat src_rgb = cv::imread("assert/01_7_additional.jpg", CV_LOAD_IMAGE_COLOR); cv::cvtColor(src_rgb, src_rgb, cv::COLOR_BGR2RGB);
    cv::Mat dst_rgb = cv::imread("assert/06_1_front.jpg", CV_LOAD_IMAGE_COLOR); cv::cvtColor(dst_rgb, dst_rgb, cv::COLOR_BGR2RGB);
    // openpose
    std::vector<cv::Mat> src_candidate_subset = openpose.call(src_rgb);
    std::vector<cv::Mat> dst_candidate_subset = openpose.call(dst_rgb);
    cv::Mat src_candidate = src_candidate_subset[0];
    cv::Mat src_subset = src_candidate_subset[1];
    cv::Mat dst_candidate = dst_candidate_subset[0];
    cv::Mat dst_subset = dst_candidate_subset[1];
    // lipjpp
    cv::Mat src_seg = lipjpp.call(src_rgb);
    cv::Mat dst_seg = lipjpp.call(dst_rgb);
    // densepose
    cv::Mat src_iuv = densepose.call(src_rgb);
    // ct
    cv::Mat fake = ct.call(src_rgb, dst_rgb, src_candidate, src_subset, dst_candidate, dst_subset, src_seg, dst_seg, src_iuv);




    cv::cvtColor(fake, fake, cv::COLOR_RGB2BGR);
    cv::imwrite("fake.png", fake);


    

    return 0;
}
