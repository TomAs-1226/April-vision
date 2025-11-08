#include "FrameProcessor.h"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

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
FrameProcessor::FrameProcessor()
    : tagSizeM_(config::DEFAULT_TAG_SIZE_M)
    , useCLAHE_(true)
    , gamma_(1.25)
    , baseDecimate_(config::DEFAULT_DECIMATE)
    , useWhitelist_(false)
    , useBlacklist_(false)
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

void FrameProcessor::setGamma(double gamma) {
    if (std::abs(gamma - gamma_) > 1e-6) {
        gamma_ = gamma;
        buildGammaLUT(gamma);
    }
}

void FrameProcessor::buildGammaLUT(double gamma) {
    gammaLUT_.resize(256);
    if (std::abs(gamma - 1.0) < 1e-3) {
        for (int i = 0; i < 256; i++) {
            gammaLUT_[i] = i;
        }
    } else {
        double invGamma = 1.0 / gamma;
        for (int i = 0; i < 256; i++) {
            gammaLUT_[i] = static_cast<uint8_t>(
                std::pow(i / 255.0, invGamma) * 255.0
            );
        }
    }
}

cv::Mat FrameProcessor::preprocessImage(const cv::Mat& gray) {
    cv::Mat result;

    if (useCLAHE_) {
        clahe_->apply(gray, result);

        if (std::abs(gamma_ - 1.0) > 1e-3) {
            cv::LUT(result, gammaLUT_, result);
        }
    } else {
        result = gray.clone();
    }

    return result;
}

double FrameProcessor::computeBlurVariance(const cv::Mat& gray) {
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    return stddev[0] * stddev[0];
}

int FrameProcessor::chooseDecimate(int base, double blurVar) {
    if (blurVar < config::BLUR_HIGH) {
        return std::max(base, config::ADAPT_DECIMATE_HIGH);
    } else if (blurVar < config::BLUR_MED) {
        return std::max(base, config::ADAPT_DECIMATE_MED);
    } else {
        return std::max(base, config::ADAPT_DECIMATE_LOW);
    }
}

bool FrameProcessor::shouldProcessTag(int id) {
    if (useWhitelist_ && whitelist_.find(id) == whitelist_.end()) {
        return false;
    }
    if (useBlacklist_ && blacklist_.find(id) != blacklist_.end()) {
        return false;
    }
    return true;
}

cv::Mat FrameProcessor::processFrame(const cv::Mat& frame, ProcessingStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (frame.empty()) {
        return cv::Mat();
    }

    int h = frame.rows;
    int w = frame.cols;

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

    // Adaptive decimation based on blur
    double blurVar = computeBlurVariance(grayProc);
    int decimate = chooseDecimate(baseDecimate_, blurVar);
    detector_->setDecimate(static_cast<float>(decimate));

    // Detect tags
    std::vector<Detection> detections = detector_->detect(grayProc);

    // Filter detections
    std::vector<Detection> filtered;
    for (const auto& det : detections) {
        if (shouldProcessTag(det.id)) {
            filtered.push_back(det);
        }
    }

    // Create visualization (grayscale base with color overlays)
    cv::Mat vis;
    cv::cvtColor(grayProc, vis, cv::COLOR_GRAY2BGR);

    std::set<int> visibleIds;
    std::vector<TagData> tagDataList;

    // Process each detection
    for (auto& det : filtered) {
        visibleIds.insert(det.id);

        // Refine corners if area is large enough
        std::vector<cv::Point2f> corners = det.corners;
        double area = PoseEstimator::polygonArea(corners);

        if (area > 30.0) {
            cv::cornerSubPix(
                grayProc, corners,
                cv::Size(config::CORNER_SUBPIX_WIN, config::CORNER_SUBPIX_WIN),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                               config::CORNER_SUBPIX_MAX_ITER,
                               config::CORNER_SUBPIX_EPS)
            );
        }

        // Update tracker
        updateTracker(det.id, corners);
        lastCorners_[det.id] = corners;

        // Compute Limelight values
        double tx_deg, ty_deg, ta_percent;
        PoseEstimator::computeLimelightValues(corners, cameraMatrix_, w, h,
                                            tx_deg, ty_deg, ta_percent);

        // Solve pose (only for reasonably large tags)
        double diag = cv::norm(corners[0] - corners[2]);
        cv::Vec3d tvec(0, 0, 0), rvec(0, 0, 0);
        double reprojErr = 0.0;

        if (diag >= 14.0) {
            auto poseResult = poseEstimator_->solvePose(corners, tagSizeM_,
                                                       cameraMatrix_, distCoeffs_);
            if (poseResult) {
                // Smooth position
                auto& posSmooth = posSmoothers_[det.id];
                if (!posSmooth || std::abs(posSmooth->getAlpha() - emaPosAlpha_) > 1e-9) {
                    posSmooth = std::make_unique<EMASmoother>(emaPosAlpha_);
                }

                Eigen::Vector3d tvecEigen(poseResult->tvec[0], poseResult->tvec[1], poseResult->tvec[2]);
                Eigen::VectorXd smoothedPos = posSmooth->update(tvecEigen);

                // Smooth orientation
                auto& poseSmooth = poseSmoothers_[det.id];
                if (!poseSmooth || std::abs(poseSmooth->getAlpha() - emaPoseAlpha_) > 1e-9) {
                    poseSmooth = std::make_unique<EMASmoother>(emaPoseAlpha_);
                }

                Eigen::Vector3d rvecEigen(poseResult->rvec[0], poseResult->rvec[1], poseResult->rvec[2]);
                Eigen::VectorXd smoothedRot = poseSmooth->update(rvecEigen);

                // Apply median filter if enabled
                if (config::POSE_MEDIAN_WINDOW > 1) {
                    auto& medBuf = poseMedians_[det.id];
                    if (!medBuf) {
                        medBuf = std::make_unique<MedianBuffer>(config::POSE_MEDIAN_WINDOW);
                    }

                    Eigen::VectorXd combined(6);
                    combined << smoothedPos, smoothedRot;
                    medBuf->push(combined);

                    auto medResult = medBuf->median();
                    if (medResult) {
                        tvec = cv::Vec3d((*medResult)(0), (*medResult)(1), (*medResult)(2));
                        rvec = cv::Vec3d((*medResult)(3), (*medResult)(4), (*medResult)(5));
                    } else {
                        tvec = cv::Vec3d(smoothedPos(0), smoothedPos(1), smoothedPos(2));
                        rvec = cv::Vec3d(smoothedRot(0), smoothedRot(1), smoothedRot(2));
                    }
                } else {
                    tvec = cv::Vec3d(smoothedPos(0), smoothedPos(1), smoothedPos(2));
                    rvec = cv::Vec3d(smoothedRot(0), smoothedRot(1), smoothedRot(2));
                }

                reprojErr = poseResult->reprojError;

                // Add to network data
                TagData tagData;
                tagData.id = det.id;
                tagData.tx_deg = tx_deg;
                tagData.ty_deg = ty_deg;
                tagData.ta_percent = ta_percent;
                tagData.tvec = tvec;
                tagData.rvec = rvec;
                tagData.reprojError = reprojErr;
                tagDataList.push_back(tagData);
            }
        }

        // Draw detection
        drawDetection(vis, det, corners, tx_deg, ty_deg, ta_percent, tvec);
    }

    // Handle invisible tags (prediction with optical flow + Kalman)
    if (!prevGray_.empty()) {
        predictInvisibleTags(vis, w, h, visibleIds, prevGray_, gray);
    }

    // Scene purge logic
    if (visibleIds.empty()) {
        if (!sceneHasUnseen_) {
            sceneUnseenStart_ = std::chrono::steady_clock::now();
            sceneHasUnseen_ = true;
        } else {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - sceneUnseenStart_
            ).count();

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

    // Publish to network
    if (publisher_ && !tagDataList.empty()) {
        VisionPayload payload;
        payload.timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        payload.tags = tagDataList;
        publisher_->publish(payload);
    }

    // Update stats
    auto t1 = std::chrono::high_resolution_clock::now();
    double procTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    processTimeHist_.push_back(procTimeMs);
    if (processTimeHist_.size() > 30) {
        processTimeHist_.pop_front();
    }

    // Adaptive rate control
    if (processTimeHist_.size() == 30) {
        double avgMs = 0.0;
        for (double t : processTimeHist_) avgMs += t;
        avgMs /= processTimeHist_.size();

        if (avgMs > config::PROCESS_TIME_HIGH_MS && detectionRate_ > config::MIN_DET_RATE) {
            detectionRate_ = std::max(config::MIN_DET_RATE, detectionRate_ * 0.85);
        } else if (avgMs < config::PROCESS_TIME_LOW_MS && detectionRate_ < config::MAX_DET_RATE) {
            detectionRate_ = std::min(config::MAX_DET_RATE, detectionRate_ * 1.08);
        }

        processTimeHist_.clear();
    }

    stats.detectionRateHz = detectionRate_;
    stats.avgProcessTimeMs = procTimeMs;
    stats.tagCount = static_cast<int>(filtered.size());
    stats.blurVariance = blurVar;

    return vis;
}

void FrameProcessor::updateTracker(int id, const std::vector<cv::Point2f>& corners) {
    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) {
        cx += pt.x;
        cy += pt.y;
    }
    cx /= corners.size();
    cy /= corners.size();

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
    for (auto& [id, tracker] : trackers_) {
        if (visibleIds.find(id) != visibleIds.end()) {
            continue;
        }

        double unseenTime = tracker->secondsUnseen();

        // Remove if too old
        if (unseenTime > config::REMOVAL_TIMEOUT) {
            continue;
        }

        // Skip if in fade-out period
        if (unseenTime > config::KEEPALIVE_TIMEOUT) {
            tracker->predict();
            continue;
        }

        // Try optical flow first
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
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i]) goodCount++;
            }

            if (goodCount >= 3 && p1.size() >= 4) {
                // Clamp points to image bounds
                for (auto& pt : p1) {
                    pt.x = std::max(1.0f, std::min(pt.x, static_cast<float>(width - 2)));
                    pt.y = std::max(1.0f, std::min(pt.y, static_cast<float>(height - 2)));
                }

                // Update tracker with optical flow result
                double cx = 0.0, cy = 0.0;
                for (const auto& pt : p1) {
                    cx += pt.x;
                    cy += pt.y;
                }
                cx /= p1.size();
                cy /= p1.size();

                double diag = cv::norm(p1[0] - p1[2]);
                diag = std::clamp(diag, config::MIN_SCALE_PX, config::MAX_SCALE_PX);

                tracker->update(cx, cy, diag);
                lkLastPts_[id] = p1;
                lastCorners_[id] = p1;

                // Draw optical flow prediction
                std::vector<cv::Point> pts;
                for (const auto& pt : p1) {
                    pts.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
                }
                cv::polylines(vis, pts, true, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);

                drewOpticalFlow = true;
            }
        }

        // Fall back to Kalman prediction
        if (!drewOpticalFlow) {
            tracker->predict();
            auto state = tracker->get();

            if (state) {
                auto [cx, cy, s] = *state;

                // Limit prediction distance
                auto lastCornersIt = lastCorners_.find(id);
                if (lastCornersIt != lastCorners_.end()) {
                    double lastCx = 0.0, lastCy = 0.0;
                    for (const auto& pt : lastCornersIt->second) {
                        lastCx += pt.x;
                        lastCy += pt.y;
                    }
                    lastCx /= lastCornersIt->second.size();
                    lastCy /= lastCornersIt->second.size();

                    double dist = std::hypot(cx - lastCx, cy - lastCy);
                    double maxMove = config::MAX_PREDICT_DISTANCE * std::hypot(width, height);

                    if (dist > maxMove) {
                        double factor = maxMove / (dist + 1e-6);
                        cx = lastCx + (cx - lastCx) * factor;
                        cy = lastCy + (cy - lastCy) * factor;
                    }
                }

                // Clamp to extended bounds
                int pad = std::min(width, height) / 2;
                cx = std::clamp(cx, -static_cast<double>(pad), static_cast<double>(width - 1 + pad));
                cy = std::clamp(cy, -static_cast<double>(pad), static_cast<double>(height - 1 + pad));
                s = std::clamp(s, config::MIN_SCALE_PX, config::MAX_SCALE_PX);

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
    // Draw polygon
    std::vector<cv::Point> pts;
    for (const auto& pt : corners) {
        pts.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
    }
    cv::polylines(vis, pts, true, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);

    // Draw text overlay
    double cx = 0.0, cy = 0.0;
    for (const auto& pt : corners) {
        cx += pt.x;
        cy += pt.y;
    }
    cx /= corners.size();
    cy /= corners.size();

    std::ostringstream oss;
    oss << "ID" << det.id << " tx=" << std::fixed << std::setprecision(1) << tx
        << "° ty=" << ty << "° ta=" << std::setprecision(2) << ta
        << "% Z=" << std::setprecision(3) << tvec[2] << "m";

    std::string text = oss.str();
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 2, &baseline);

    int x0 = std::max(2, static_cast<int>(cx - textSize.width / 2));
    int y0 = std::max(2, static_cast<int>(cy - textSize.height - 8));

    x0 = std::min(x0, vis.cols - textSize.width - 2);
    y0 = std::min(y0, vis.rows - textSize.height - 2);

    cv::rectangle(vis,
                 cv::Point(x0 - 4, y0 - 4),
                 cv::Point(x0 + textSize.width + 4, y0 + textSize.height + 6),
                 cv::Scalar(0, 0, 0), -1);

    cv::putText(vis, text, cv::Point(x0, y0 + textSize.height),
               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // Draw ID near corner
    std::string idText = "id=" + std::to_string(det.id);
    int tx_text = static_cast<int>(corners[0].x);
    int ty_text = static_cast<int>(corners[0].y - 20);

    cv::Size idSize = cv::getTextSize(idText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(vis,
                 cv::Point(tx_text - 3, ty_text - idSize.height - 3),
                 cv::Point(tx_text + idSize.width + 3, ty_text + 3),
                 cv::Scalar(0, 0, 0), -1);

    cv::putText(vis, idText, cv::Point(tx_text, ty_text),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void FrameProcessor::drawPrediction(cv::Mat& vis, int id, double cx, double cy, double s, bool isOpticalFlow) {
    int half = std::max(6, static_cast<int>(s * 0.5));
    cv::Point p1(static_cast<int>(cx - half), static_cast<int>(cy - half));
    cv::Point p2(static_cast<int>(cx + half), static_cast<int>(cy + half));

    cv::Scalar color = isOpticalFlow ? cv::Scalar(0, 180, 255) : cv::Scalar(0, 160, 255);
    cv::rectangle(vis, p1, p2, color, 2, cv::LINE_AA);
}