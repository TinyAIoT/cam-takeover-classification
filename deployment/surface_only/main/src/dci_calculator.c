// Dynamic Comfort Index calculation based on method from:
//      Bíl, M., Andrášik, R., & Kubeček, J. (2015). How comfortable are your cycling tracks? 
//      A new method for objective bicycle vibration measurement. 
//      Transportation Research Part C: Emerging Technologies, 56, 415–425. 
//      https://doi.org/10.1016/j.trc.2015.05.007


#include "dci_calculator.h"
#include "esp_log.h"
#include <math.h>


static const char* TAG = "DCI";
#define DCI_ACCEL_LSB_TO_G(sensitivity) (1.0f / (sensitivity))  // Dynamic based on range

// Calculate gravity reference from IMU data (assuming some stationary periods)
esp_err_t dci_calculate_gravity_reference(const imu_data_t* imu_samples, int count, float* gravity_ref) {
    if (!imu_samples || !gravity_ref || count < DCI_MIN_SAMPLES) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Simple approach: Use the mean magnitude of acceleration vectors
    float sum_magnitude = 0.0f;
    int valid_samples = 0;
    
    for (int i = 0; i < count; i++) {
        if (!imu_samples[i].valid) continue;
        
        // Convert to g-force and calculate magnitude
        float ax = imu_samples[i].accel_x * DCI_ACCEL_LSB_TO_G(imu_get_accel_sensitivity());
        float ay = imu_samples[i].accel_y * DCI_ACCEL_LSB_TO_G(imu_get_accel_sensitivity());
        float az = imu_samples[i].accel_z * DCI_ACCEL_LSB_TO_G(imu_get_accel_sensitivity());

        float magnitude = sqrtf(ax*ax + ay*ay + az*az);
        sum_magnitude += magnitude;
        valid_samples++;
    }
    
    if (valid_samples < DCI_MIN_SAMPLES) {
        ESP_LOGE(TAG, "Not enough valid samples for gravity reference");
        return ESP_ERR_INVALID_STATE;
    }
    
    *gravity_ref = sum_magnitude / valid_samples;
    
    ESP_LOGD(TAG, "Calculated gravity reference: %.3f g (from %d samples)", 
             *gravity_ref, valid_samples);
    
    return ESP_OK;
}

// Calculate DCI according to Bill et al. equation
esp_err_t dci_calculate(const imu_data_t* imu_samples, int count, float gravity_ref, dci_result_t* result) {
    if (!imu_samples || !result || count < DCI_MIN_SAMPLES) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Initialize result
    memset(result, 0, sizeof(dci_result_t));
    result->gravity_reference = gravity_ref;
    
    float sum_squared_vertical = 0.0f;
    int above_threshold_count = 0;
    int valid_samples = 0;
    
    ESP_LOGD(TAG, "Processing %d samples with gravity reference %.3f g", count, gravity_ref);
    
    for (int i = 0; i < count; i++) {
        if (!imu_samples[i].valid) continue;

        // Convert X-axis acceleration to g-force (horizontal acceleration)
        float ax_g = imu_samples[i].accel_x * DCI_ACCEL_LSB_TO_G(imu_get_accel_sensitivity());

        // Remove gravity bias and get absolute vertical acceleration
        // float vertical_accel = fabsf(az_g) - gravity_ref;
        
        // Check if above 1g threshold
        if (fabsf(ax_g) > DCI_GRAVITY_THRESHOLD) {
            sum_squared_vertical += ax_g * ax_g;
            above_threshold_count++;
            ESP_LOGD(TAG, "Sample %d: ax=%.3f g (above threshold)", 
                     i, ax_g);
        }
        
        valid_samples++;
    }
    
    result->total_samples = valid_samples;
    result->above_threshold = above_threshold_count;
    
    // Calculate DCI according to equation: DCI = (1/n * Σ(avi²))^(-1)
    if (above_threshold_count >= DCI_MIN_SAMPLES) {
        float mean_squared = sum_squared_vertical / above_threshold_count;
        result->dci_value = 1.0f / mean_squared;  // DCI = (mean)^(-1)
        result->valid = true;

        ESP_LOGD(TAG, "DCI calculated: %.3f (from %d/%d samples above threshold (1g))", 
                 result->dci_value, above_threshold_count, valid_samples);
    } else {
        result->dci_value = 0.0f;
        result->valid = false;

        ESP_LOGD(TAG, "DCI invalid: only %d samples above threshold (1g) (need ≥%d)", 
                 above_threshold_count, DCI_MIN_SAMPLES);
    }
    
    return ESP_OK;
}

// Convenience function that calculates gravity reference and DCI in one call
esp_err_t dci_calculate_with_auto_gravity(const imu_data_t* imu_samples, int count, dci_result_t* result) {
    if (!imu_samples || !result) {
        return ESP_ERR_INVALID_ARG;
    }
    
    float gravity_ref;
    esp_err_t ret = dci_calculate_gravity_reference(imu_samples, count, &gravity_ref);
    if (ret != ESP_OK) {
        return ret;
    }
    
    return dci_calculate(imu_samples, count, gravity_ref, result);
}