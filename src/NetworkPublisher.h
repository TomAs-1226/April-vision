#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <opencv2/opencv.hpp>
#include <optional>

#ifdef APRILV_HAS_NTCORE
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
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
};

struct VisionPayload {
    double timestamp;
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

    void setNtServer(const std::string& server);
    std::string ntServer() const { return ntServer_; }

    void enableNetworkTables(bool enable) { ntEnabled_ = enable; }
    void enableUDP(bool enable) { udpEnabled_ = enable; }

    bool isNetworkTablesEnabled() const { return ntEnabled_; }
    bool isUDPEnabled() const { return udpEnabled_; }

private:
    void publishLoop();
    bool initUDP();
    void sendUDP(const std::string& json);
    void ensureNetworkTables();
    void publishNetworkTables(const VisionPayload& payload);

    std::string ntServer_;
    std::string udpIp_;
    int udpPort_;

    std::atomic<bool> running_;
    std::atomic<bool> ntEnabled_;
    std::atomic<bool> udpEnabled_;

    std::thread publishThread_;
    std::mutex queueMutex_;
    std::deque<VisionPayload> queue_;

    mutable std::mutex ntMutex_;

    // UDP socket
    int udpSocket_;
    struct sockaddr_in udpAddr_;
    bool udpInitialized_;

    static constexpr size_t MAX_QUEUE_SIZE = 8;

#ifdef APRILV_HAS_NTCORE
    nt::NetworkTableInstance ntInstance_;
    std::shared_ptr<nt::NetworkTable> table_;
    std::optional<uint64_t> connectionListener_;
    bool ntReady_{false};
#endif
};