#include "imu.h"
#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "IMU";

// ICM-20948 registers
#define ICM20948_WHO_AM_I       0x00
#define ICM20948_PWR_MGMT_1     0x06
#define ICM20948_PWR_MGMT_2     0x07
#define ICM20948_ACCEL_XOUT_H   0x2D
#define ICM20948_GYRO_XOUT_H    0x33
#define ICM20948_TEMP_OUT_H     0x39
#define ICM20948_WHO_AM_I_VAL   0xEA

// where to adjust the measurement range (±2g, ±4g, ±8g, ±16g) and corresponding sensitivity?
#define ICM20948_ACCEL_CONFIG    0x14
#define ICM20948_ACCEL_FS_SEL_2G  0x00
#define ICM20948_ACCEL_FS_SEL_4G  0x08
#define ICM20948_ACCEL_FS_SEL_8G  0x10
#define ICM20948_ACCEL_FS_SEL_16G 0x18

// I2C handles
static i2c_master_bus_handle_t bus_handle = NULL;
static i2c_master_dev_handle_t dev_handle = NULL;
static bool imu_ready = false;

// Write single register
static esp_err_t write_reg(uint8_t reg, uint8_t val) {
    uint8_t data[2] = {reg, val};
    return i2c_master_transmit(dev_handle, data, 2, pdMS_TO_TICKS(100));
}

// Read register(s)
static esp_err_t read_reg(uint8_t reg, uint8_t* data, size_t len) {
    return i2c_master_transmit_receive(dev_handle, &reg, 1, data, len, pdMS_TO_TICKS(100));
}

esp_err_t imu_init(void) {
    ESP_LOGI(TAG, "Initializing IMU...");
    
    // Create I2C bus
    i2c_master_bus_config_t bus_config = {
        .i2c_port = IMU_I2C_PORT,
        .sda_io_num = IMU_SDA_PIN,
        .scl_io_num = IMU_SCL_PIN,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    
    esp_err_t ret = i2c_new_master_bus(&bus_config, &bus_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2C bus: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Add device
    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = IMU_I2C_ADDR,
        .scl_speed_hz = IMU_I2C_FREQ,
    };
    
    ret = i2c_master_bus_add_device(bus_handle, &dev_config, &dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add I2C device: %s", esp_err_to_name(ret));
        i2c_del_master_bus(bus_handle);
        return ret;
    }
    
    // Check WHO_AM_I
    uint8_t who_am_i;
    ret = read_reg(ICM20948_WHO_AM_I, &who_am_i, 1);
    if (ret != ESP_OK || who_am_i != ICM20948_WHO_AM_I_VAL) {
        ESP_LOGE(TAG, "WHO_AM_I check failed: 0x%02X (expected 0x%02X)", who_am_i, ICM20948_WHO_AM_I_VAL);
        imu_deinit();
        return ESP_FAIL;
    }
    
    // Reset device
    write_reg(ICM20948_PWR_MGMT_1, 0x80);
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Wake up and enable sensors
    write_reg(ICM20948_PWR_MGMT_1, 0x01); // Auto select clock
    write_reg(ICM20948_PWR_MGMT_2, 0x00); // Enable accel + gyro
    write_reg(ICM20948_ACCEL_CONFIG, ICM20948_ACCEL_FS_SEL_8G); // Set accel range to ±8g
    
    imu_ready = true;
    ESP_LOGI(TAG, "IMU initialized successfully");
    return ESP_OK;
}

esp_err_t imu_read(imu_data_t* data) {
    if (!imu_ready || !data) {
        return ESP_ERR_INVALID_STATE;
    }
    
    uint8_t raw_data[20]; // Enough for all sensors
    esp_err_t ret;
    
    data->valid = false;
    
    // Read accelerometer (6 bytes)
    ret = read_reg(ICM20948_ACCEL_XOUT_H, raw_data, 6);
    if (ret == ESP_OK) {
        data->accel_x = (raw_data[0] << 8) | raw_data[1];
        data->accel_y = (raw_data[2] << 8) | raw_data[3];
        data->accel_z = (raw_data[4] << 8) | raw_data[5];
    } else {
        ESP_LOGW(TAG, "Failed to read accelerometer");
        return ret;
    }
    
    // Read gyroscope (6 bytes)
    ret = read_reg(ICM20948_GYRO_XOUT_H, raw_data, 6);
    if (ret == ESP_OK) {
        data->gyro_x = (raw_data[0] << 8) | raw_data[1];
        data->gyro_y = (raw_data[2] << 8) | raw_data[3];
        data->gyro_z = (raw_data[4] << 8) | raw_data[5];
    } else {
        ESP_LOGW(TAG, "Failed to read gyroscope");
        return ret;
    }
    
    // Read temperature (2 bytes)
    ret = read_reg(ICM20948_TEMP_OUT_H, raw_data, 2);
    if (ret == ESP_OK) {
        data->temp = (raw_data[0] << 8) | raw_data[1];
    } else {
        ESP_LOGW(TAG, "Failed to read temperature");
        return ret;
    }
    
    // Magnetometer is more complex, skip for now
    data->mag_x = data->mag_y = data->mag_z = 0;
    
    data->valid = true;
    return ESP_OK;
}

void imu_deinit(void) {
    if (dev_handle) {
        i2c_master_bus_rm_device(dev_handle);
        dev_handle = NULL;
    }
    if (bus_handle) {
        i2c_del_master_bus(bus_handle);
        bus_handle = NULL;
    }
    imu_ready = false;
    ESP_LOGI(TAG, "IMU deinitialized");
}