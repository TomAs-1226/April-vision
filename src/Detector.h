#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
}

struct Detection {
    int id{};
    std::vector<cv::Point2f> corners;  // p0..p3
    double decision_margin{};
};

class Detector {
public:
    Detector();
    ~Detector();

    // Expects CV_8UC1 (we enforce grayscale/contiguity inside detect()).
    std::vector<Detection> detect(const cv::Mat& img);
    std::vector<Detection> detectROI(const cv::Mat& img, const cv::Rect& roi);

    void setDecimate(float v);
    void setBlur(float v);
    void setRefineEdges(bool on);
    void setNumThreads(int n);

private:
    apriltag_detector_t* td_{nullptr};
    apriltag_family_t*   tf_{nullptr};

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
};
