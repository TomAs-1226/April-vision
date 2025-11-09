#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Refines points in-place, but only for corners that are safely inside the image.
// Returns number of refined points.
inline int safeCornerSubPix(cv::Mat src,
                            std::vector<cv::Point2f>& pts,
                            cv::Size winSize = {5,5},
                            cv::Size zeroZone = {-1,-1},
                            cv::TermCriteria criteria = cv::TermCriteria(
                                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01))
{
    if (src.empty() || pts.empty()) return 0;

    // cornerSubPix expects single-channel 8U/32F; convert on the fly if needed.
    if (src.type() != CV_8UC1 && src.type() != CV_32FC1) {
        cv::Mat tmp;
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
        src = tmp;
    }

    // Build a "safe ROI": points must be at least winSize/2 away from borders.
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

    // This call will assert if any point is outside; our filter prevents that.
    cv::cornerSubPix(src, safePts, winSize, zeroZone, criteria);  // OpenCV requirement. :contentReference[oaicite:2]{index=2}

    for (int k = 0; k < (int)safePts.size(); ++k)
        pts[idxMap[k]] = safePts[k];

    return (int)safePts.size();
}

// Utility: quick check if any corner is near a border (in pixels).
inline bool anyCornerNearBorder(const std::vector<cv::Point2f>& c,
                                int width, int height, int marginPx)
{
    for (const auto& p : c)
        if (p.x < marginPx || p.y < marginPx ||
            p.x >= width - marginPx || p.y >= height - marginPx)
            return true;
    return false;
}
