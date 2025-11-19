#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class ArucoPoseHelper {
public:
    ArucoPoseHelper();

    bool estimatePose(const std::vector<cv::Point2f>& corners,
                      double tagSizeMeters,
                      const cv::Mat& cameraMatrix,
                      const cv::Mat& distCoeffs,
                      cv::Vec3d& rvec,
                      cv::Vec3d& tvec,
                      double& reprojError);

private:
    std::vector<cv::Point3f> objPoints_;
};
