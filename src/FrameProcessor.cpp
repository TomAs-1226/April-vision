#include "FrameProcessor.h"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

// Undefine Windows macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------- Safe SubPix helpers (local-only; no extra header needed) --------
namespace {
    // Returns true if any corner lies within 'margin' pixels of the border.
    inline bool anyCornerNearBorder(const std::vector<cv::Point2f>& c,
                                    int width, int height, int marginPx)
    {
        for (const auto& p : c) {
            if (p.x < marginPx || p.y < marginPx ||
                p.x >= width - marginPx || p.y >= height - marginPx)
                return true;
        }
        return false;
    }

    // Refines points in-place on 'src' but only for points whose sub-pix window
    // fits fully inside the image. This prevents the OpenCV assertion:
    // Rect(0,0,src.cols,src.rows).contains(cT) in cv::cornerSubPix   (imgproc/cornersubpix.cpp:99)
    // (OpenCV requires every refined point to be strictly inside the image.)  :contentReference[oaicite:2]{index=2}
    inline int safeCornerSubPix(cv::Mat src,
                                std::vector<cv::Point2f>& pts,
                                cv::Size winSize,
                                cv::Size zeroZone = {-1,-1},
                                cv::TermCriteria criteria = cv::TermCriteria(
                                    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01))
    {
        if (src.empty() || pts.empty()) return 0;

        // cornerSubPix requires 8U/32F single-channel
        if (src.type() != CV_8UC1 && src.type() != CV_32FC1) {
            cv::Mat tmp;
            cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
            src = tmp;
        }

        const int rx = winSize.width  / 2;
        const int ry = winSize.height / 2;
        const cv::Rect safeROI(rx, ry,
                               std::max(0, src.cols - 2*rx),
                               std::max(0, src.rows - 2*ry));

        std::vector<cv::Point2f> safePts;
        safePts.reserve(pts.size());
        std::vector<int> idxMap; idxMap.reserve(pts.size());

        for (int i = 0; i < (int)pts.size(); ++i) {
            if (safeROI.contains(pts[i])) {
                safePts.push_back(pts[i]);
                idxMap.push_back(i);
            }
        }

        if (safePts.empty()) return 0;

        cv::cornerSubPix(src, safePts, winSize, zeroZone, criteria);

        for (int k = 0; k < (int)safePts.size(); ++k)
            pts[idxMap[k]] = safePts[k];

        return (int)safePts.size();
    }
} // namespace
// --------------------------------------------------------------------------


FrameProcessor::FrameProcessor()
    : tagSizeM_(config::DEFAULT_TAG_SIZE_M)
    , useCLAHE_(true)
    , gamma_(1.25)
    , baseDecimate_(config::DEFAULT_DECIMATE)
    , useWhitelist_(false)
    , useBlacklist_(false)
    , highSpeedMode_(false)
    , useROITracking_(false)
    , activeROI_()
    , roiMissCount_(0)
    , userCLAHEPref_(true)
    , userGamma_(1.25)
    , userDecimate_(config::DEFAULT_DECIMATE)
    , sceneHasUnseen_(false)
    , detectionRate_(config::DETECTION_RATE_HZ)
    , emaPosAlpha_(config::EMA_ALPHA_POS)
    , emaPoseAlpha_(config::EMA_ALPHA_POSE)
{
    detector_ = std::make_unique<Detector>();
    poseEstimator_ = std::make_unique<PoseEstimator>();

    clahe_ = cv::createCLAHE(config::CLAHE_CLIP_LIMIT,
                             cv::Size(config::CLAHE_TILE_SIZE, config::CLAHE_TILE_SIZE));

    buildGammaLUT(gamma_);

    userCLAHEPref_ = useCLAHE_;
    userGamma_ = gamma_;
    userDecimate_ = baseDecimate_;

    std::cout << "[FrameProcessor] Initialized" << std::endl;
}

FrameProcessor::~FrameProcessor() = default;

void FrameProcessor::setCameraMatrix(const cv::Mat& K, const cv::Mat& D) {
    K.copyTo(cameraMatrix_);
    D.copyTo(distCoeffs_);
}

void FrameProcessor::setEMAAlpha(double pos, double pose) {
    emaPosAlpha_ = pos;
    emaPoseAlpha_ = pose;
}

void FrameProcessor::setDecimate(int dec) {
    userDecimate_ = std::max(1, dec);
    if (!highSpeedMode_) {
        baseDecimate_ = userDecimate_;
    }
}

void FrameProcessor::enableCLAHE(bool enable) {
    userCLAHEPref_ = enable;
    if (!highSpeedMode_) {
        useCLAHE_ = enable;
    }
}

void FrameProcessor::setGamma(double gamma) {
    userGamma_ = gamma;
    if (!highSpeedMode_) {
        applyGammaInternal(gamma);
    }
}

void FrameProcessor::applyGammaInternal(double gamma) {
    if (std::abs(gamma - gamma_) > 1e-6) {
        gamma_ = gamma;
        buildGammaLUT(gamma);
    }
}

void FrameProcessor::setHighSpeedMode(bool enable) {
    if (highSpeedMode_ == enable) return;
    highSpeedMode_ = enable;

    if (highSpeedMode_) {
        useROITracking_ = true;
        useCLAHE_ = false;
        applyGammaInternal(config::HIGH_SPEED_GAMMA);
        baseDecimate_ = std::max(1, config::HIGH_SPEED_DECIMATE);
        if (detector_) detector_->setRefineEdges(false);
        resetROI();
    } else {
        useROITracking_ = false;
        useCLAHE_ = userCLAHEPref_;
        applyGammaInternal(userGamma_);
        baseDecimate_ = userDecimate_;
        if (detector_) detector_->setRefineEdges(true);
        resetROI();
    }
}

void FrameProcessor::buildGammaLUT(double gamma) {
    gammaLUT_.resize(256);
    if (std::abs(gamma - 1.0) < 1e-3) {
        for (int i = 0; i < 256; i++) gammaLUT_[i] = (uint8_t)i;
    } else {
        const double inv = 1.0 / gamma;
        for (int i = 0; i < 256; i++) {
            gammaLUT_[i] = static_cast<uint8_t>(cv::saturate_cast<uchar>(std::pow(i / 255.0, inv) * 255.0));
        }
    }
}

cv::Mat FrameProcessor::preprocessImage(const cv::Mat& gray) {
    if (preprocessBuf_.empty() || preprocessBuf_.size() != gray.size()) {
        preprocessBuf_.create(gray.size(), gray.type());
    }

    if (useCLAHE_) {
        clahe_->apply(gray, preprocessBuf_);
    } else {
        gray.copyTo(preprocessBuf_);
    }

    if (std::abs(gamma_ - 1.0) > 1e-3) {
        cv::LUT(preprocessBuf_, gammaLUT_, preprocessBuf_);
    }

    return preprocessBuf_;
}

double FrameProcessor::computeBlurVariance(const cv::Mat& gray) {
    cv::Mat lap; cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev; cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

int FrameProcessor::chooseDecimate(int base, double blurVar) {
    int decimate = std::max(1, base);
    if      (blurVar < config::BLUR_HIGH) decimate = std::min(decimate, config::ADAPT_DECIMATE_LOW);
    else if (blurVar < config::BLUR_MED)  decimate = std::min(decimate, config::ADAPT_DECIMATE_MED);
    else                                  decimate = std::min(decimate, config::ADAPT_DECIMATE_HIGH);
    return std::max(1, decimate);
}

bool FrameProcessor::shouldProcessTag(int id) {
    if (useWhitelist_ && whitelist_.find(id) == whitelist_.end()) return false;
    if (useBlacklist_ && blacklist_.find(id) != blacklist_.end()) return false;
    return true;
}

void FrameProcessor::resetROI() {
    activeROI_ = cv::Rect();
    roiMissCount_ = 0;
}

cv::Rect FrameProcessor::expandROI(const cv::Rect& base, int width, int height) const {
    if (base.width <= 0 || base.height <= 0) return cv::Rect();

    const int margin = config::HIGH_SPEED_ROI_MARGIN_PX;
    const double grow = config::HIGH_SPEED_ROI_EXPAND_PCT;
    const int extraX = margin + static_cast<int>(std::round(base.width * grow));
    const int extraY = margin + static_cast<int>(std::round(base.height * grow));

    int x = std::max(0, base.x - extraX);
    int y = std::max(0, base.y - extraY);
    int w = std::min(width - x, base.width + 2 * extraX);
    int h = std::min(height - y, base.height + 2 * extraY);

    w = std::max(w, std::min(width, config::HIGH_SPEED_ROI_MIN_W));
    h = std::max(h, std::min(height, config::HIGH_SPEED_ROI_MIN_H));

    if (x + w > width) w = width - x;
    if (y + h > height) h = height - y;

    return cv::Rect(x, y, std::max(1, w), std::max(1, h));
}

cv::Rect FrameProcessor::computeHistoryROI(int width, int height) const {
    if (lastCorners_.empty()) return cv::Rect();

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    bool hasPoint = false;

    for (const auto& kv : lastCorners_) {
        for (const auto& pt : kv.second) {
            minX = std::min(minX, pt.x);
            minY = std::min(minY, pt.y);
            maxX = std::max(maxX, pt.x);
            maxY = std::max(maxY, pt.y);
            hasPoint = true;
        }
    }

    if (!hasPoint) return cv::Rect();

    const int x0 = std::max(0, static_cast<int>(std::floor(minX)));
    const int y0 = std::max(0, static_cast<int>(std::floor(minY)));
    const int x1 = std::min(width, static_cast<int>(std::ceil(maxX)));
    const int y1 = std::min(height, static_cast<int>(std::ceil(maxY)));
    cv::Rect base(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
    return expandROI(base, width, height);
}

cv::Mat FrameProcessor::processFrame(const cv::Mat& frame, ProcessingStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (frame.empty()) {
        return cv::Mat();
    }

    const int h = frame.rows;
    const int w = frame.cols;

    // Initialize camera matrix if needed
    if (cameraMatrix_.empty()) {
        PoseEstimator::defaultCameraMatrix(w, h, cameraMatrix_, distCoeffs_);
    }

    // Convert to grayscale (reuse buffer)
    if (grayBuf_.size() != frame.size()) {
        grayBuf_ = cv::Mat(h, w, CV_8UC1);
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, grayBuf_, cv::COLOR_BGR2GRAY);
        gray = grayBuf_;
    } else {
        gray = frame;
    }

    // Preprocess
    cv::Mat grayProc = preprocessImage(gray);
    if (!grayProc.isContinuous()) grayProc = grayProc.clone();  // stride safety

    cv::Rect roiToUse(0, 0, w, h);
    bool usingROI = false;
    if (useROITracking_) {
        cv::Rect candidate = activeROI_.area() > 0 ? activeROI_ : computeHistoryROI(w, h);
        if (candidate.width > 0 && candidate.height > 0 && candidate.area() < w * h) {
            roiToUse = candidate;
            usingROI = true;
        }
    }

    // Adaptive decimation based on blur → handed to AprilTag (internal decimate)
    const double blurVar = computeBlurVariance(grayProc);
    const int decimate = chooseDecimate(baseDecimate_, blurVar);
    detector_->setDecimate(static_cast<float>(decimate));

    cv::Mat detectInput;
    if (usingROI) {
        cv::Mat roiView(grayProc, roiToUse);
        if (roiBuf_.empty() || roiBuf_.size() != roiToUse.size()) {
            roiBuf_.create(roiToUse.size(), grayProc.type());
        }
        roiView.copyTo(roiBuf_);
        detectInput = roiBuf_;
    } else {
        detectInput = grayProc;
    }

    // Detect tags ON THE SAME IMAGE we'll refine on
    std::vector<Detection> detections = detector_->detect(detectInput);

    if (usingROI) {
        const cv::Point2f offset(static_cast<float>(roiToUse.x), static_cast<float>(roiToUse.y));
        for (auto& det : detections) {
            for (auto& pt : det.corners) {
                pt += offset;
            }
        }
    }

    // Filter detections
    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    for (const auto& det : detections) {
        if (shouldProcessTag(det.id)) filtered.push_back(det);
    }

    // Visualization
    cv::Mat vis;
    if (frame.channels() == 3) vis = frame.clone();
    else                       cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);

    std::set<int> visibleIds;
    std::vector<TagData> tagDataList;
    float roiMinX = std::numeric_limits<float>::max();
    float roiMinY = std::numeric_limits<float>::max();
    float roiMaxX = std::numeric_limits<float>::lowest();
    float roiMaxY = std::numeric_limits<float>::lowest();
    bool roiHasData = false;

    // Process each detection
    for (auto& det : filtered) {
        visibleIds.insert(det.id);

        // Order corners & compute polygon area
        std::vector<cv::Point2f> corners = PoseEstimator::orderCorners(det.corners);
        const double area = PoseEstimator::polygonArea(corners);

        // ---- Safe sub-pixel refinement on the SAME image/scale used for detection ----
        // Prevents: OpenCV(…) error: Rect(0,0,src.cols,src.rows).contains(cT) in cv::cornerSubPix
        // (points must be inside the image; we skip those too close to borders). :contentReference[oaicite:3]{index=3}
        if (area > 30.0) {
            const cv::Size subWin(config::CORNER_SUBPIX_WIN, config::CORNER_SUBPIX_WIN);
            const int margin = std::max(subWin.width, subWin.height) / 2;

            if (!anyCornerNearBorder(corners, grayProc.cols, grayProc.rows, margin)) {
                safeCornerSubPix(grayProc, corners, subWin);
            }
            // else: skip refinement for this tag this frame (too close to edge)
        }

        // Update tracker
        updateTracker(det.id, corners);
        lastCorners_[det.id] = corners;

        if (useROITracking_) {
            for (const auto& pt : corners) {
                roiMinX = std::min(roiMinX, pt.x);
                roiMinY = std::min(roiMinY, pt.y);
                roiMaxX = std::max(roiMaxX, pt.x);
                roiMaxY = std::max(roiMaxY, pt.y);
            }
            roiHasData = true;
        }

        // Compute Limelight-style values
        double tx_deg, ty_deg, ta_percent;
        PoseEstimator::computeLimelightValues(corners, cameraMatrix_, w, h,
                                              tx_deg, ty_deg, ta_percent);

        // Solve pose when large enough
        const double diag = cv::norm(corners[0] - corners[2]);
        cv::Vec3d tvec(0, 0, 0), rvec(0, 0, 0);
        double reprojErr = 0.0;

        if (diag >= 14.0) {
            auto poseResult = poseEstimator_->solvePose(corners, tagSizeM_,
                                                        cameraMatrix_, distCoeffs_);
            if (poseResult) {
                // Smooth position
                auto& posSmooth = posSmoothers_[det.id];
                if (!posSmooth || std::abs(posSmooth->getAlpha() - emaPosAlpha_) > 1e-9)
                    posSmooth = std::make_unique<EMASmoother>(emaPosAlpha_);
                Eigen::Vector3d tvecE(poseResult->tvec[0], poseResult->tvec[1], poseResult->tvec[2]);
                Eigen::VectorXd sPos = posSmooth->update(tvecE);

                // Smooth orientation
                auto& poseSmooth = poseSmoothers_[det.id];
                if (!poseSmooth || std::abs(poseSmooth->getAlpha() - emaPoseAlpha_) > 1e-9)
                    poseSmooth = std::make_unique<EMASmoother>(emaPoseAlpha_);
                Eigen::Vector3d rvecE(poseResult->rvec[0], poseResult->rvec[1], poseResult->rvec[2]);
                Eigen::VectorXd sRot = poseSmooth->update(rvecE);

                // Median filter if enabled
                if (config::POSE_MEDIAN_WINDOW > 1) {
                    auto& med = poseMedians_[det.id];
                    if (!med) med = std::make_unique<MedianBuffer>(config::POSE_MEDIAN_WINDOW);
                    Eigen::VectorXd both(6); both << sPos, sRot;
                    med->push(both);
                    if (auto m = med->median()) {
                        tvec = cv::Vec3d((*m)(0), (*m)(1), (*m)(2));
                        rvec = cv::Vec3d((*m)(3), (*m)(4), (*m)(5));
                    } else {
                        tvec = cv::Vec3d(sPos(0), sPos(1), sPos(2));
                        rvec = cv::Vec3d(sRot(0), sRot(1), sRot(2));
                    }
                } else {
                    tvec = cv::Vec3d(sPos(0), sPos(1), sPos(2));
                    rvec = cv::Vec3d(sRot(0), sRot(1), sRot(2));
                }

                reprojErr = poseResult->reprojError;

                TagData td;
                td.id = det.id;
                td.tx_deg = tx_deg; td.ty_deg = ty_deg; td.ta_percent = ta_percent;
                td.tvec = tvec; td.rvec = rvec; td.reprojError = reprojErr;
                tagDataList.push_back(td);
            }
        }

        // Draw detection
        drawDetection(vis, det, corners, tx_deg, ty_deg, ta_percent, tvec);
    }

    if (useROITracking_) {
        if (roiHasData) {
            const int x0 = std::max(0, static_cast<int>(std::floor(roiMinX)));
            const int y0 = std::max(0, static_cast<int>(std::floor(roiMinY)));
            const int x1 = std::min(w, static_cast<int>(std::ceil(roiMaxX)));
            const int y1 = std::min(h, static_cast<int>(std::ceil(roiMaxY)));
            cv::Rect base(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
            activeROI_ = expandROI(base, w, h);
            roiMissCount_ = 0;
        } else if (activeROI_.area() > 0) {
            ++roiMissCount_;
            if (roiMissCount_ >= config::HIGH_SPEED_ROI_MAX_MISSES) {
                resetROI();
            } else {
                activeROI_ = expandROI(activeROI_, w, h);
            }
        }
    }

    // Predict invisible tags (optical flow + Kalman)
    if (!highSpeedMode_ && !prevGray_.empty()) {
        predictInvisibleTags(vis, w, h, visibleIds, prevGray_, gray);
    }

    // Scene purge logic
    if (visibleIds.empty()) {
        if (!sceneHasUnseen_) {
            sceneUnseenStart_ = std::chrono::steady_clock::now();
            sceneHasUnseen_ = true;
        } else {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - sceneUnseenStart_).count();
            if (elapsed >= config::SCENE_PURGE_TIMEOUT) {
                trackers_.clear();
                posSmoothers_.clear();
                poseSmoothers_.clear();
                poseMedians_.clear();
                lastCorners_.clear();
                lkLastPts_.clear();
                sceneHasUnseen_ = false;
            }
        }
    } else {
        sceneHasUnseen_ = false;
    }

    // Purge old trackers
    purgeOldTrackers(visibleIds);

    // Store current frame for optical flow
    gray.copyTo(prevGray_);

    // Publish
    if (publisher_ && !tagDataList.empty()) {
        VisionPayload payload;
        payload.timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        payload.tags = tagDataList;
        publisher_->publish(payload);
    }

    // Stats
    auto t1 = std::chrono::high_resolution_clock::now();
    const double procTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    processTimeHist_.push_back(procTimeMs);
    if (processTimeHist_.size() > 30) processTimeHist_.pop_front();

    if (processTimeHist_.size() == 30) {
        double avgMs = 0.0; for (double t : processTimeHist_) avgMs += t;
        avgMs /= processTimeHist_.size();
        const double minRate = static_cast<double>(config::MIN_DET_RATE);
        const double maxRate = static_cast<double>(config::MAX_DET_RATE);

        if (avgMs > config::PROCESS_TIME_HIGH_MS && detectionRate_ > minRate)
            detectionRate_ = std::max(minRate, detectionRate_ * 0.85);
        else if (avgMs < config::PROCESS_TIME_LOW_MS && detectionRate_ < maxRate)
            detectionRate_ = std::min(maxRate, detectionRate_ * 1.08);

        processTimeHist_.clear();
    }

    stats.detectionRateHz = detectionRate_;
    stats.avgProcessTimeMs = procTimeMs;
    stats.frameTimeMs = procTimeMs;
    stats.tagCount = static_cast<int>(filtered.size());
    stats.blurVariance = blurVar;
    stats.highSpeedMode = highSpeedMode_;
    stats.usedROI = usingROI;
    stats.roiRect = usingROI ? roiToUse : cv::Rect();

    return vis;
}

void FrameProcessor::updateTracker(int id, const std::vector<cv::Point2f>& corners) {
    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) { cx += pt.x; cy += pt.y; }
    cx /= corners.size(); cy /= corners.size();

    double diag = cv::norm(corners[0] - corners[2]);
    diag = std::max(config::MIN_SCALE_PX, diag);

    auto& tracker = trackers_[id];
    if (!tracker) {
        tracker = std::make_unique<BoxTracker>(config::TRACK_Q, config::TRACK_R);
        tracker->init(cx, cy, diag);
    } else {
        tracker->update(cx, cy, diag);
    }

    lkLastPts_[id] = corners;
}

void FrameProcessor::predictInvisibleTags(cv::Mat& vis, int width, int height,
                                         const std::set<int>& visibleIds,
                                         const cv::Mat& grayPrev, const cv::Mat& grayCurr)
{
    for (auto it = trackers_.begin(); it != trackers_.end(); ++it) {
        int id = it->first;
        auto& tracker = it->second;

        if (visibleIds.find(id) != visibleIds.end()) continue;

        double unseenTime = tracker->secondsUnseen();

        // Remove if too old
        if (unseenTime > config::REMOVAL_TIMEOUT) continue;

        // Skip if in fade-out period
        if (unseenTime > config::KEEPALIVE_TIMEOUT) {
            tracker->predict();
            continue;
        }

        // Try optical flow
        bool drewOpticalFlow = false;
        auto lkIt = lkLastPts_.find(id);

        if (lkIt != lkLastPts_.end() && !lkIt->second.empty()) {
            std::vector<cv::Point2f> p1;
            std::vector<uchar> status;
            std::vector<float> err;

            cv::calcOpticalFlowPyrLK(
                grayPrev, grayCurr, lkIt->second, p1, status, err,
                cv::Size(config::LK_WIN_SIZE, config::LK_WIN_SIZE),
                config::LK_MAX_LEVEL,
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                 config::LK_MAX_ITER, config::LK_EPSILON)
            );

            int goodCount = 0;
            for (size_t i = 0; i < status.size(); i++) if (status[i]) goodCount++;

            if (goodCount >= 3 && p1.size() >= 4) {
                // Clamp to image bounds defensively
                for (auto& pt : p1) {
                    pt.x = std::max(1.0f, std::min(pt.x, static_cast<float>(width  - 2)));
                    pt.y = std::max(1.0f, std::min(pt.y, static_cast<float>(height - 2)));
                }

                // Update tracker with OF result
                double cx = 0.0, cy = 0.0;
                for (const auto& pt : p1) { cx += pt.x; cy += pt.y; }
                cx /= p1.size(); cy /= p1.size();

                double diag = cv::norm(p1[0] - p1[2]);
                diag = std::clamp(diag, config::MIN_SCALE_PX, config::MAX_SCALE_PX);

                tracker->update(cx, cy, diag);
                lkLastPts_[id] = p1;
                lastCorners_[id] = p1;

                // Draw OF prediction
                std::vector<cv::Point> pts;
                pts.reserve(p1.size());
                for (const auto& pt : p1)
                    pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                cv::polylines(vis, pts, true, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);

                drewOpticalFlow = true;
            }
        }

        // Fall back to Kalman-only prediction
        if (!drewOpticalFlow) {
            tracker->predict();
            if (auto state = tracker->get()) {
                auto [cx, cy, s] = *state;

                // Limit prediction distance vs last known
                auto lastIt = lastCorners_.find(id);
                if (lastIt != lastCorners_.end()) {
                    double lastCx = 0.0, lastCy = 0.0;
                    for (const auto& pt : lastIt->second) { lastCx += pt.x; lastCy += pt.y; }
                    lastCx /= lastIt->second.size(); lastCy /= lastIt->second.size();

                    const double dist = std::hypot(cx - lastCx, cy - lastCy);
                    const double maxMove = config::MAX_PREDICT_DISTANCE * std::hypot(width, height);
                    if (dist > maxMove) {
                        const double f = maxMove / (dist + 1e-6);
                        cx = lastCx + (cx - lastCx) * f;
                        cy = lastCy + (cy - lastCy) * f;
                    }
                }

                // Clamp to extended bounds
                const int pad = std::min(width, height) / 2;
                cx = std::clamp(cx, -static_cast<double>(pad), static_cast<double>(width  - 1 + pad));
                cy = std::clamp(cy, -static_cast<double>(pad), static_cast<double>(height - 1 + pad));
                s  = std::clamp(s,  config::MIN_SCALE_PX,       config::MAX_SCALE_PX);

                drawPrediction(vis, id, cx, cy, s, false);
            }
        }
    }
}

void FrameProcessor::purgeOldTrackers(const std::set<int>& visibleIds) {
    std::vector<int> toRemove;
    for (const auto& [id, tracker] : trackers_) {
        if (visibleIds.find(id) == visibleIds.end()) {
            if (tracker->secondsUnseen() > config::REMOVAL_TIMEOUT) {
                toRemove.push_back(id);
            }
        }
    }
    for (int id : toRemove) {
        trackers_.erase(id);
        posSmoothers_.erase(id);
        poseSmoothers_.erase(id);
        poseMedians_.erase(id);
        lastCorners_.erase(id);
        lkLastPts_.erase(id);
    }
}

void FrameProcessor::drawDetection(cv::Mat& vis, const Detection& det,
                                  const std::vector<cv::Point2f>& corners,
                                  double tx, double ty, double ta, const cv::Vec3d& tvec)
{
    std::vector<cv::Point> pts; pts.reserve(corners.size());
    for (const auto& pt : corners)
        pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
    cv::polylines(vis, pts, true, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);

    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) { cx += pt.x; cy += pt.y; }
    cx /= corners.size(); cy /= corners.size();

    std::ostringstream oss;
    oss << "ID" << det.id
        << " tx=" << std::fixed << std::setprecision(1) << tx
        << "° ty=" << ty
        << "° ta=" << std::setprecision(2) << ta << "%"
        << " Z="  << std::setprecision(3) << tvec[2] << "m";

    std::string text = oss.str();
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 2, &baseline);

    int x0 = std::max(2, static_cast<int>(cx - textSize.width / 2));
    int y0 = std::max(2, static_cast<int>(cy - textSize.height - 8));
    x0 = std::min(x0, vis.cols - textSize.width - 2);
    y0 = std::min(y0, vis.rows - textSize.height - 2);

    cv::rectangle(vis, {x0 - 4, y0 - 4}, {x0 + textSize.width + 4, y0 + textSize.height + 6},
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(vis, text, {x0, y0 + textSize.height},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    const std::string idText = "id=" + std::to_string(det.id);
    int tx_text = static_cast<int>(corners[0].x);
    int ty_text = static_cast<int>(corners[0].y - 20);
    const cv::Size idSize = cv::getTextSize(idText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::rectangle(vis,
                  {tx_text - 3, ty_text - idSize.height - 3},
                  {tx_text + idSize.width + 3, ty_text + 3},
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(vis, idText, {tx_text, ty_text},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void FrameProcessor::drawPrediction(cv::Mat& vis, int id, double cx, double cy, double s, bool isOpticalFlow) {
    const int half = std::max(6, static_cast<int>(s * 0.5));
    const cv::Point p1(static_cast<int>(cx - half), static_cast<int>(cy - half));
    const cv::Point p2(static_cast<int>(cx + half), static_cast<int>(cy + half));
    const cv::Scalar color = isOpticalFlow ? cv::Scalar(0, 180, 255) : cv::Scalar(0, 160, 255);
    cv::rectangle(vis, p1, p2, color, 2, cv::LINE_AA);
}
