#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <opencv2/opencv.hpp>
#include <array>

#ifdef USE_NTCORE
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <networktables/NetworkTableEntry.h>
#endif

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

struct TagData {
    int id;
    double tx_deg;
    double ty_deg;
    double ta_percent;
    cv::Vec3d tvec;
    cv::Vec3d rvec;
    double reprojError;
    std::array<cv::Point2f, 4> corners;
    double skewDeg;
    double shortSidePx;
    double longSidePx;
    double boundingWidthPx;
    double boundingHeightPx;
    double areaPx;
    double poseAmbiguity;
    double decisionMargin;
};

struct VisionPayload {
    double timestamp;
    double pipelineLatencyMs;
    std::vector<TagData> tags;
};

class NetworkPublisher {
public:
    NetworkPublisher(const std::string& ntServer,
                    const std::string& udpIp,
                    int udpPort);
    ~NetworkPublisher();

    void start();
    void stop();
    void publish(const VisionPayload& payload);

    void enableNetworkTables(bool enable) { ntEnabled_ = enable; }
    void enableUDP(bool enable) { udpEnabled_ = enable; }

    bool isNetworkTablesEnabled() const { return ntEnabled_; }
    bool isUDPEnabled() const { return udpEnabled_; }

private:
    void publishLoop();
    bool initUDP();
    void sendUDP(const std::string& json);
    void publishNetworkTables(const VisionPayload& payload);

#ifdef USE_NTCORE
    void configureNetworkTables();
#endif

    std::string ntServer_;
    std::string udpIp_;
    int udpPort_;

    std::atomic<bool> running_;
    std::atomic<bool> ntEnabled_;
    std::atomic<bool> udpEnabled_;

    std::thread publishThread_;
    std::mutex queueMutex_;
    std::deque<VisionPayload> queue_;

    // UDP socket
    int udpSocket_;
    struct sockaddr_in udpAddr_;
    bool udpInitialized_;

#ifdef USE_NTCORE
    nt::NetworkTableInstance ntInstance_;
    std::shared_ptr<nt::NetworkTable> visionTable_;
    std::shared_ptr<nt::NetworkTable> limelightTable_;
    nt::NetworkTableEntry timestampEntry_;
    nt::NetworkTableEntry latencyEntry_;
    nt::NetworkTableEntry idsEntry_;
    nt::NetworkTableEntry txEntry_;
    nt::NetworkTableEntry tyEntry_;
    nt::NetworkTableEntry taEntry_;
    nt::NetworkTableEntry xyzEntry_;
    nt::NetworkTableEntry rpyEntry_;
    nt::NetworkTableEntry distanceEntry_;
    nt::NetworkTableEntry bestIdEntry_;
    nt::NetworkTableEntry bestPoseEntry_;
    nt::NetworkTableEntry bestRpyEntry_;
    nt::NetworkTableEntry bestDistanceEntry_;
    nt::NetworkTableEntry connectedEntry_;
    nt::NetworkTableEntry ambiguityEntry_;
    nt::NetworkTableEntry llTvEntry_;
    nt::NetworkTableEntry llTidEntry_;
    nt::NetworkTableEntry llTsEntry_;
    nt::NetworkTableEntry llTlEntry_;
    nt::NetworkTableEntry llTshortEntry_;
    nt::NetworkTableEntry llTlongEntry_;
    nt::NetworkTableEntry llThorEntry_;
    nt::NetworkTableEntry llTvertEntry_;
    nt::NetworkTableEntry llPoseAmbEntry_;
    nt::NetworkTableEntry llTargetPoseCamEntry_;
    nt::NetworkTableEntry llTargetPoseRobotEntry_;
    nt::NetworkTableEntry llCameraPoseRobotEntry_;
    nt::NetworkTableEntry llCameraPoseTargetEntry_;
    nt::NetworkTableEntry llBotPoseTargetEntry_;
#endif

#ifdef USE_NTCORE
    cv::Matx33d camToRobotR_;
    cv::Matx33d robotToCamR_;
    cv::Vec3d camToRobotT_;
    cv::Vec3d robotToCamT_;
    bool ntConfigured_ = false;
#endif

    static constexpr size_t MAX_QUEUE_SIZE = 8;
};