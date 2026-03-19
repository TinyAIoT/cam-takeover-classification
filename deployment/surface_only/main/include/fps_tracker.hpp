#ifndef FPS_TRACKER_HPP
#define FPS_TRACKER_HPP

#include <cstdint>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

class FPSTracker {
private:
    static constexpr int WINDOW_SIZE = 50;
    uint32_t last_time;
    float sum;
    float values[WINDOW_SIZE];
    int index;
    int count;
    const char* tag;  // For logging identification
    
public:
    FPSTracker(const char* log_tag = "FPS");
    
    void update();
    float get_average() const;
    float get_current() const;
    int get_sample_count() const;
    void reset();
};

#endif // FPS_TRACKER_HPP