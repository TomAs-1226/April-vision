#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <optional>
#include <deque>

struct PoseResult {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    double reprojError;
};

class EMASmoother {
public:
    explicit EMASmoother(double alpha = 0.3);

    Eigen::VectorXd update(const Eigen::VectorXd& val);
    void reset();

    void setAlpha(double alpha) { alpha_ = alpha; }
    double getAlpha() const { return alpha_; }

private:
    double alpha_;
    Eigen::VectorXd v_;
    bool initialized_;
};

class MedianBuffer {
public:
    explicit MedianBuffer(int window);

    void push(const Eigen::VectorXd& val);
    std::optional<Eigen::VectorXd> median();
    void clear();

private:
    int window_;
    std::deque<Eigen::VectorXd> q_;
};

class PoseEstimator {
public:
    PoseEstimator();

    // Solve PnP for tag pose
    std::optional<PoseResult> solvePose(
        const std::vector<cv::Point2f>& corners,
        double tagSizeMeters,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs
    );

    // Camera calibration utilities
    static void defaultCameraMatrix(int width, int height, cv::Mat& K, cv::Mat& D);

    // Compute Limelight-style values
    static void computeLimelightValues(
        const std::vector<cv::Point2f>& corners,
        const cv::Mat& K,
        int width, int height,
        double& tx_deg, double& ty_deg, double& ta_percent
    );

    static double polygonArea(const std::vector<cv::Point2f>& pts);

    static std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point2f>& corners);
};