#include "fps_tracker.hpp"

FPSTracker::FPSTracker(const char* log_tag) 
    : last_time(0), sum(0.0f), index(0), count(0), tag(log_tag) {
    memset(values, 0, sizeof(values));
}

void FPSTracker::update() {
    uint32_t now = xTaskGetTickCount();
    if (last_time != 0) {
        float seconds = (now - last_time) * portTICK_PERIOD_MS / 1000.0f;
        if (seconds > 0.0f) {
            float fps = 1.0f / seconds;
            
            // Remove old value from sum if buffer is full
            if (count == WINDOW_SIZE) {
                sum -= values[index];
            }
            
            // Add new value
            values[index] = fps;
            sum += fps;
            
            // Update counters
            index = (index + 1) % WINDOW_SIZE;
            if (count < WINDOW_SIZE) count++;
            
            // Log current and average FPS
            float avg = sum / count;
            ESP_LOGD(tag, "Current: %.2f FPS, Avg (last %d): %.2f FPS", fps, count, avg);
        }
    }
    last_time = now;
}

float FPSTracker::get_average() const {
    return count > 0 ? sum / count : 0.0f;
}

float FPSTracker::get_current() const {
    return count > 0 ? values[(index - 1 + WINDOW_SIZE) % WINDOW_SIZE] : 0.0f;
}

int FPSTracker::get_sample_count() const {
    return count;
}

void FPSTracker::reset() {
    last_time = 0;
    sum = 0.0f;
    index = 0;
    count = 0;
    memset(values, 0, sizeof(values));
}