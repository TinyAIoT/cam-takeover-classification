#include "imu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include <string.h>

static const char* TAG = "IMU";

// Global state
static icm20948_device_t icm_device;
static icm20948_config_i2c_t icm_config;
static bool imu_ready = false;
static icm20948_accel_config_fs_sel_e current_accel_range = GPM_8;
static icm20948_gyro_config_1_fs_sel_e current_gyro_range = DPS_2000;

// Sensitivity values for different ranges (LSB per unit)
static const float accel_sensitivity[] = {16384.0f, 8192.0f, 4096.0f, 2048.0f}; // ±2g, ±4g, ±8g, ±16g
static const float gyro_sensitivity[] = {131.0f, 65.5f, 32.8f, 16.4f};          // ±250, ±500, ±1000, ±2000 dps

esp_err_t imu_init(void) {
    ESP_LOGI(TAG, "Initializing IMU with ICM20948 library...");
    
    // Setup I2C configuration
    icm_config.i2c_port = IMU_I2C_PORT;
    icm_config.i2c_addr = IMU_I2C_ADDR;
    icm_config.scl_speed_hz = IMU_I2C_FREQ;
    icm_config.bus_handle = NULL;
    icm_config.dev_handle = NULL;
    
    // Initialize ICM20948 with new I2C driver
    icm20948_status_e status = icm20948_init_i2c_new(&icm_device, &icm_config);
    if (status != ICM_20948_STAT_OK) {
        ESP_LOGE(TAG, "Failed to initialize ICM20948 I2C interface");
        return ESP_FAIL;
    }
    
    // Verify device identity
    uint8_t whoami;
    status = icm20948_get_who_am_i(&icm_device, &whoami);
    if (status != ICM_20948_STAT_OK || whoami != ICM_20948_WHOAMI) {
        ESP_LOGE(TAG, "WHO_AM_I check failed: 0x%02X (expected 0x%02X)", whoami, ICM_20948_WHOAMI);
        icm20948_deinit_i2c_new(&icm_config);
        return ESP_FAIL;
    }
    
    // Software reset
    ESP_LOGI(TAG, "Performing software reset");
    status = icm20948_sw_reset(&icm_device);
    if (status != ICM_20948_STAT_OK) {
        ESP_LOGE(TAG, "Software reset failed");
        icm20948_deinit_i2c_new(&icm_config);
        return ESP_FAIL;
    }
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Wake up device
    icm20948_sleep(&icm_device, false);
    icm20948_low_power(&icm_device, false);
    
    // Set clock source to auto-select
    icm20948_set_clock_source(&icm_device, CLOCK_AUTO);
    
    // Configure sensors
    icm20948_internal_sensor_id_bm sensors = 
        (icm20948_internal_sensor_id_bm)(ICM_20948_INTERNAL_ACC | ICM_20948_INTERNAL_GYR);
    
    // Set sample mode to continuous
    status = icm20948_set_sample_mode(&icm_device, sensors, SAMPLE_MODE_CONTINUOUS);
    if (status != ICM_20948_STAT_OK) {
        ESP_LOGE(TAG, "Failed to set sample mode");
        icm20948_deinit_i2c_new(&icm_config);
        return ESP_FAIL;
    }
    
    // Set full scale ranges (±8g for accel, ±2000dps for gyro - good for bike vibration detection)
    icm20948_fss_t fss = {
        .a = current_accel_range,  // ±8g for better dynamic range
        .g = current_gyro_range    // ±2000 dps
    };
    status = icm20948_set_full_scale(&icm_device, sensors, fss);
    if (status != ICM_20948_STAT_OK) {
        ESP_LOGE(TAG, "Failed to set full scale ranges");
        icm20948_deinit_i2c_new(&icm_config);
        return ESP_FAIL;
    }
    
    // Disable DLPF for maximum responsiveness (good for vibration detection)
    icm20948_enable_dlpf(&icm_device, ICM_20948_INTERNAL_ACC, false);
    icm20948_enable_dlpf(&icm_device, ICM_20948_INTERNAL_GYR, false);
    
    // Enable magnetometer if needed
    icm20948_i2c_master_enable(&icm_device, true);
    
    imu_ready = true;
    ESP_LOGI(TAG, "IMU initialized successfully with ±%dg accel, ±%ddps gyro", 
             (2 << current_accel_range), (250 << current_gyro_range));
    
    return ESP_OK;
}

esp_err_t imu_read(imu_data_t* data) {
    if (!imu_ready || !data) {
        return ESP_ERR_INVALID_STATE;
    }
    
    data->valid = false;
    data->timestamp_us = esp_timer_get_time();
    
    // Read all sensor data using ICM20948 library
    icm20948_agmt_t agmt;
    icm20948_status_e status = icm20948_get_agmt(&icm_device, &agmt);
    
    if (status != ICM_20948_STAT_OK) {
        ESP_LOGW(TAG, "Failed to read IMU data");
        return ESP_ERR_INVALID_RESPONSE;
    }
    
    // Copy data to our structure
    data->accel_x = agmt.acc.axes.x;
    data->accel_y = agmt.acc.axes.y;
    data->accel_z = agmt.acc.axes.z;

    // ESP_LOGD(TAG, "Accel: X=%.2f g, Y=%.2f g, Z=%.2f g", 
    //          data->accel_x * (1.0f / (imu_get_accel_sensitivity())), 
    //          data->accel_y * (1.0f / (imu_get_accel_sensitivity())), 
    //          data->accel_z * (1.0f / (imu_get_accel_sensitivity())));
    
    data->gyro_x = agmt.gyr.axes.x;
    data->gyro_y = agmt.gyr.axes.y;
    data->gyro_z = agmt.gyr.axes.z;
    
    data->mag_x = agmt.mag.axes.x;
    data->mag_y = agmt.mag.axes.y;
    data->mag_z = agmt.mag.axes.z;
    
    data->temp = agmt.tmp.val;
    data->valid = true;
    
    return ESP_OK;
}

void imu_deinit(void) {
    if (imu_ready) {
        icm20948_sleep(&icm_device, true);
        icm20948_deinit_i2c_new(&icm_config);
        imu_ready = false;
        ESP_LOGI(TAG, "IMU deinitialized");
    }
}

esp_err_t imu_set_accel_range(icm20948_accel_config_fs_sel_e range) {
    if (!imu_ready) return ESP_ERR_INVALID_STATE;
    
    icm20948_fss_t fss = {
        .a = range,
        .g = current_gyro_range
    };
    
    icm20948_status_e status = icm20948_set_full_scale(&icm_device, ICM_20948_INTERNAL_ACC, fss);
    if (status == ICM_20948_STAT_OK) {
        current_accel_range = range;
        ESP_LOGI(TAG, "Accelerometer range set to ±%dg", (2 << range));
        return ESP_OK;
    }
    
    return ESP_FAIL;
}

esp_err_t imu_set_gyro_range(icm20948_gyro_config_1_fs_sel_e range) {
    if (!imu_ready) return ESP_ERR_INVALID_STATE;
    
    icm20948_fss_t fss = {
        .a = current_accel_range,
        .g = range
    };
    
    icm20948_status_e status = icm20948_set_full_scale(&icm_device, ICM_20948_INTERNAL_GYR, fss);
    if (status == ICM_20948_STAT_OK) {
        current_gyro_range = range;
        ESP_LOGI(TAG, "Gyroscope range set to ±%ddps", (250 << range));
        return ESP_OK;
    }
    
    return ESP_FAIL;
}

esp_err_t imu_enable_dlpf(bool enable) {
    if (!imu_ready) return ESP_ERR_INVALID_STATE;
    
    icm20948_status_e status1 = icm20948_enable_dlpf(&icm_device, ICM_20948_INTERNAL_ACC, enable);
    icm20948_status_e status2 = icm20948_enable_dlpf(&icm_device, ICM_20948_INTERNAL_GYR, enable);
    
    if (status1 == ICM_20948_STAT_OK && status2 == ICM_20948_STAT_OK) {
        ESP_LOGI(TAG, "DLPF %s", enable ? "enabled" : "disabled");
        return ESP_OK;
    }
    
    return ESP_FAIL;
}

esp_err_t imu_reset(void) {
    if (!imu_ready) return ESP_ERR_INVALID_STATE;
    
    icm20948_status_e status = icm20948_sw_reset(&icm_device);
    if (status == ICM_20948_STAT_OK) {
        vTaskDelay(pdMS_TO_TICKS(100));
        ESP_LOGI(TAG, "IMU reset complete");
        return ESP_OK;
    }
    
    return ESP_FAIL;
}

float imu_get_accel_sensitivity(void) {
    return accel_sensitivity[current_accel_range];
}

float imu_get_gyro_sensitivity(void) {
    return gyro_sensitivity[current_gyro_range];
}