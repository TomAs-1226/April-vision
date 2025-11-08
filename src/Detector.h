#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
}

struct Detection {
    int id;
    std::vector<cv::Point2f> corners;
    double decision_margin;
};

class Detector {
public:
    Detector();
    ~Detector();

    // Detect AprilTags in grayscale image
    std::vector<Detection> detect(const cv::Mat& gray);

    // Configure detection parameters
    void setDecimate(float decimate);
    void setBlur(float blur);
    void setRefineEdges(bool enable);
    void setNumThreads(int threads);

private:
    apriltag_detector_t* td_;
    apriltag_family_t* tf_;

    // Prevent copying
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
};