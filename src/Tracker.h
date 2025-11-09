#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <optional>

class BoxTracker {
public:
    BoxTracker(double q = 0.02, double r = 4.0);

    void init(double cx, double cy, double s);
    Eigen::VectorXd predict();
    Eigen::VectorXd update(double cx, double cy, double s);

    std::optional<std::tuple<double, double, double>> get() const;
    double secondsUnseen() const;
    void resetUnseenTimer();

private:
    double q_;  // Process noise
    double r_;  // Measurement noise

    Eigen::VectorXd x_;  // State [cx, cy, s, vx, vy, vs]
    Eigen::MatrixXd P_;  // Covariance
    Eigen::MatrixXd F_;  // State transition
    Eigen::MatrixXd H_;  // Measurement matrix
    Eigen::MatrixXd Q_;  // Process noise covariance
    Eigen::MatrixXd R_;  // Measurement noise covariance

    std::chrono::steady_clock::time_point lastTs_;
    double unseenCount_;
    bool initialized_;
};