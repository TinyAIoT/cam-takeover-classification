#ifndef IMU_H
#define IMU_H

#include "driver/i2c_master.h"
#include "esp_log.h"
#include "icm20948.h"
#include "icm20948_i2c_new.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// IMU pins and config
#define IMU_SDA_PIN     2
#define IMU_SCL_PIN     1
#define IMU_I2C_PORT    I2C_NUM_0
#define IMU_I2C_ADDR    0x68
#define IMU_I2C_FREQ    400000

typedef struct {
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x, gyro_y, gyro_z;
    int16_t mag_x, mag_y, mag_z;
    int16_t temp;
    bool valid;
    uint64_t timestamp_us;  // Add timestamp for better data tracking
} imu_data_t;

// Simple API
esp_err_t imu_init(void);
esp_err_t imu_read(imu_data_t* data);
void imu_deinit(void);

// Configuration functions using ICM20948 capabilities
esp_err_t imu_set_accel_range(icm20948_accel_config_fs_sel_e range);
esp_err_t imu_set_gyro_range(icm20948_gyro_config_1_fs_sel_e range);
esp_err_t imu_enable_dlpf(bool enable);
esp_err_t imu_reset(void);

// Get current sensitivity values (useful for DCI calculation)
float imu_get_accel_sensitivity(void);
float imu_get_gyro_sensitivity(void);

#ifdef __cplusplus
}
#endif

#endif