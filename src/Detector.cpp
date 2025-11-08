#include "Detector.h"
#include "Config.h"
#include <iostream>
#include <cstring>

extern "C" {
    #include "apriltag.h"
    #include "tag36h11.h"
    #include "common/image_u8.h"
}

Detector::Detector() {
    // Create tag family
    tf_ = tag36h11_create();

    // Create detector
    td_ = apriltag_detector_create();
    apriltag_detector_add_family(td_, tf_);

    // Configure detector for performance
    td_->quad_decimate = config::DEFAULT_DECIMATE;
    td_->quad_sigma = 0.0f;
    td_->nthreads = 4;
    td_->refine_edges = 1;
    td_->decode_sharpening = 0.25;

    std::cout << "[Detector] Initialized with 36h11 family" << std::endl;
}

Detector::~Detector() {
    if (td_) {
        apriltag_detector_destroy(td_);
    }
    if (tf_) {
        tag36h11_destroy(tf_);
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& gray) {
    std::vector<Detection> detections;

    if (gray.empty() || gray.type() != CV_8UC1) {
        return detections;
    }

    // Create image_u8_t using image_u8_create_from_gray
    // This avoids the const member initialization issue
    image_u8_t* img = image_u8_create(gray.cols, gray.rows);
    std::memcpy(img->buf, gray.data, gray.cols * gray.rows);

    // Detect tags
    zarray_t* det_array = apriltag_detector_detect(td_, img);

    // Convert to our detection format
    for (int i = 0; i < zarray_size(det_array); i++) {
        apriltag_detection_t* det;
        zarray_get(det_array, i, &det);

        Detection d;
        d.id = det->id;
        d.decision_margin = det->decision_margin;

        // Extract corners
        d.corners.reserve(4);
        for (int j = 0; j < 4; j++) {
            d.corners.emplace_back(
                static_cast<float>(det->p[j][0]),
                static_cast<float>(det->p[j][1])
            );
        }

        detections.push_back(std::move(d));
    }

    // Cleanup
    apriltag_detections_destroy(det_array);
    image_u8_destroy(img);

    return detections;
}

void Detector::setDecimate(float decimate) {
    td_->quad_decimate = decimate;
}

void Detector::setBlur(float blur) {
    td_->quad_sigma = blur;
}

void Detector::setRefineEdges(bool enable) {
    td_->refine_edges = enable ? 1 : 0;
}

void Detector::setNumThreads(int threads) {
    td_->nthreads = threads;
}