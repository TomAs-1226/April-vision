#include "Detector.h"
#include "Config.h"
#include <thread>
#include <stdexcept>
#include <iostream>

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "common/image_u8.h"
}

Detector::Detector() {
    tf_ = tag36h11_create();
    if (!tf_) throw std::runtime_error("tag36h11_create failed");

    td_ = apriltag_detector_create();
    if (!td_) throw std::runtime_error("apriltag_detector_create failed");

    // Robust starting defaults (tune later).
    td_->quad_decimate     = static_cast<float>(config::DEFAULT_DECIMATE); // start at 1.0
    td_->quad_sigma        = 0.0f;   // try 0.8 for noisy edges
    td_->refine_edges      = 1;
    td_->decode_sharpening = 0.25f;

    int threads = config::DETECTOR_THREADS;
    if (threads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        threads = static_cast<int>(std::max<unsigned>(2, std::min<unsigned>(8, hw)));
    }
    td_->nthreads = threads;

    apriltag_detector_add_family(td_, tf_);
    std::cout << "[Detector] Initialized with 36h11 family | threads=" << td_->nthreads << std::endl;
}

Detector::~Detector() {
    if (td_) {
        apriltag_detector_remove_family(td_, tf_);
        apriltag_detector_destroy(td_);
        td_ = nullptr;
    }
    if (tf_) {
        tag36h11_destroy(tf_);
        tf_ = nullptr;
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& img_in) {
    std::vector<Detection> out;
    if (img_in.empty()) return out;

    // 1) Ensure 8-bit mono
    cv::Mat gray;
    if (img_in.type() == CV_8UC1) {
        gray = img_in;
    } else {
        cv::cvtColor(img_in, gray, cv::COLOR_BGR2GRAY);
    }

    // 2) Ensure contiguous (stride-safe)
    if (!gray.isContinuous()) gray = gray.clone();

    // 3) Allocate an image_u8_t with the same width/height/stride, then copy.
    //    We avoid brace/aggregate initialization entirely to be C++17/MSVC-safe.
    const unsigned w = static_cast<unsigned>(gray.cols);
    const unsigned h = static_cast<unsigned>(gray.rows);
    const unsigned s = static_cast<unsigned>(gray.step);   // CRITICAL: row step (stride)

    image_u8_t* im = image_u8_create_stride(w, h, s);      // official C API
    if (!im) return out;
    // Fast copy: the buffer is h*s bytes
    std::memcpy(im->buf, gray.data, static_cast<size_t>(h) * s);

    // 4) Detect
    zarray_t* dets = apriltag_detector_detect(td_, im);

    const int n = zarray_size(dets);
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        apriltag_detection_t* det = nullptr;
        zarray_get(dets, i, &det);
        if (!det) continue;

        Detection d;
        d.id = det->id;
        d.decision_margin = det->decision_margin;
        d.corners.reserve(4);
        for (int j = 0; j < 4; ++j) {
            d.corners.emplace_back(
                static_cast<float>(det->p[j][0]),
                static_cast<float>(det->p[j][1]));
        }
        out.emplace_back(std::move(d));
    }

    // 5) Cleanup
    apriltag_detections_destroy(dets);
    image_u8_destroy(im);
    return out;
}

void Detector::setDecimate(float v)   { td_->quad_decimate = v; }
void Detector::setBlur(float v)       { td_->quad_sigma    = v; }
void Detector::setRefineEdges(bool on){ td_->refine_edges  = on ? 1 : 0; }
void Detector::setNumThreads(int n)   { td_->nthreads      = n; }
