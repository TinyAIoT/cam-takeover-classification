#ifndef IMU_H
#define IMU_H

#include "driver/i2c_master.h"
#include "esp_log.h"
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

// Simple IMU data structure
typedef struct {
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x, gyro_y, gyro_z;
    int16_t mag_x, mag_y, mag_z;
    int16_t temp;
    bool valid;
} imu_data_t;

// Simple API
esp_err_t imu_init(void);
esp_err_t imu_read(imu_data_t* data);
void imu_deinit(void);

#ifdef __cplusplus
}
#endif

#endif