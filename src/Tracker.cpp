#include "Tracker.h"
#include "Config.h"
#include <algorithm>

// Undefine Windows macros
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

BoxTracker::BoxTracker(double q, double r)
    : q_(q), r_(r), unseenCount_(0.0), initialized_(false)
{
    x_ = Eigen::VectorXd::Zero(6);
    P_ = Eigen::MatrixXd::Identity(6, 6) * 16.0;
    F_ = Eigen::MatrixXd::Identity(6, 6);

    H_ = Eigen::MatrixXd::Zero(3, 6);
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;

    Q_ = Eigen::MatrixXd::Identity(6, 6) * q_;
    R_ = Eigen::MatrixXd::Identity(3, 3) * r_;

    lastTs_ = std::chrono::steady_clock::now();
}

void BoxTracker::init(double cx, double cy, double s) {
    x_ << cx, cy, s, 0.0, 0.0, 0.0;
    P_ = Eigen::MatrixXd::Identity(6, 6) * 16.0;
    lastTs_ = std::chrono::steady_clock::now();
    unseenCount_ = 0.0;
    initialized_ = true;
}

Eigen::VectorXd BoxTracker::predict() {
    if (!initialized_) {
        return x_;
    }

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - lastTs_).count();
    dt = std::max(0.001, dt);
    lastTs_ = now;

    // Update state transition matrix with dt
    F_ = Eigen::MatrixXd::Identity(6, 6);
    F_(0, 3) = dt;
    F_(1, 4) = dt;
    F_(2, 5) = dt;

    // Predict state
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;

    // Apply velocity decay
    x_(3) *= config::VELOCITY_DECAY;
    x_(4) *= config::VELOCITY_DECAY;
    x_(5) *= config::VELOCITY_DECAY;

    unseenCount_ += dt;

    return x_;
}

Eigen::VectorXd BoxTracker::update(double cx, double cy, double s) {
    if (!initialized_) {
        init(cx, cy, s);
        return x_;
    }

    // Measurement vector
    Eigen::VectorXd z(3);
    z << cx, cy, s;

    // Innovation
    Eigen::VectorXd y = z - H_ * x_;

    // Innovation covariance
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;

    // Kalman gain
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // Update state
    x_ = x_ + K * y;

    // Update covariance
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    P_ = (I - K * H_) * P_;

    unseenCount_ = 0.0;

    return x_;
}

std::optional<std::tuple<double, double, double>> BoxTracker::get() const {
    if (!initialized_) {
        return std::nullopt;
    }
    return std::make_tuple(x_(0), x_(1), x_(2));
}

double BoxTracker::secondsUnseen() const {
    return unseenCount_;
}

void BoxTracker::resetUnseenTimer() {
    unseenCount_ = 0.0;
}