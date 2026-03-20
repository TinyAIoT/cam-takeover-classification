#include "driver/i2c_master.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "icm20948.h"
#include "icm20948_i2c_new.h"

static const char* TAG = "ICM20948_I2C";

icm20948_status_e icm20948_internal_write_i2c_new(uint8_t reg, uint8_t *data, uint32_t len, void *user)
{
    if (!user) return ICM_20948_STAT_ERR;
    
    icm20948_config_i2c_t *config = (icm20948_config_i2c_t*)user;
    if (!config->dev_handle) return ICM_20948_STAT_ERR;
    
    uint8_t write_buf[len + 1];
    write_buf[0] = reg;
    memcpy(&write_buf[1], data, len);
    
    esp_err_t ret = i2c_master_transmit(config->dev_handle, write_buf, len + 1, 100 / portTICK_PERIOD_MS);
    return (ret == ESP_OK) ? ICM_20948_STAT_OK : ICM_20948_STAT_ERR;
}

icm20948_status_e icm20948_internal_read_i2c_new(uint8_t reg, uint8_t *buff, uint32_t len, void *user)
{
    if (!user || !buff) return ICM_20948_STAT_ERR;
    
    icm20948_config_i2c_t *config = (icm20948_config_i2c_t*)user;
    if (!config->dev_handle) return ICM_20948_STAT_ERR;
    
    esp_err_t ret = i2c_master_transmit_receive(config->dev_handle, &reg, 1, buff, len, pdMS_TO_TICKS(100));
    return (ret == ESP_OK) ? ICM_20948_STAT_OK : ICM_20948_STAT_ERR;
}

icm20948_status_e icm20948_init_i2c_new(icm20948_device_t *icm_device, icm20948_config_i2c_t *config)
{
    if (!icm_device || !config) {
        return ICM_20948_STAT_PARAM_ERR;
    }
    
    // Create I2C master bus
    i2c_master_bus_config_t bus_config = {
        .i2c_port = config->i2c_port,
        .sda_io_num = GPIO_NUM_2,  // Default, can be made configurable
        .scl_io_num = GPIO_NUM_1,  // Default, can be made configurable
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    
    esp_err_t ret = i2c_new_master_bus(&bus_config, &config->bus_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2C master bus: %s", esp_err_to_name(ret));
        return ICM_20948_STAT_ERR;
    }
    
    // Add I2C device
    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = config->i2c_addr,
        .scl_speed_hz = config->scl_speed_hz,
    };
    
    ret = i2c_master_bus_add_device(config->bus_handle, &dev_config, &config->dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add I2C device: %s", esp_err_to_name(ret));
        i2c_del_master_bus(config->bus_handle);
        config->bus_handle = NULL;
        return ICM_20948_STAT_ERR;
    }
    
    // Initialize ICM20948 device structure
    icm20948_init_struct(icm_device);
    
    // Setup serif with new I2C functions
    static icm20948_serif_t serif = {
        .write = icm20948_internal_write_i2c_new,
        .read = icm20948_internal_read_i2c_new,
        .user = NULL,
    };
    serif.user = (void*)config;
    
    icm20948_link_serif(icm_device, &serif);

#ifdef CONFIG_ICM_20948_USE_DMP
    icm_device->_dmp_firmware_available = true;
#else
    icm_device->_dmp_firmware_available = false;
#endif

    icm_device->_firmware_loaded = false;
    icm_device->_last_bank = 255;
    icm_device->_last_mems_bank = 255;
    icm_device->_gyroSF = 0;
    icm_device->_gyroSFpll = 0;
    icm_device->_enabled_Android_0 = 0;
    icm_device->_enabled_Android_1 = 0;
    icm_device->_enabled_Android_intr_0 = 0;
    icm_device->_enabled_Android_intr_1 = 0;

    ESP_LOGI(TAG, "ICM20948 I2C initialized successfully");
    return ICM_20948_STAT_OK;
}

void icm20948_deinit_i2c_new(icm20948_config_i2c_t *config)
{
    if (!config) return;
    
    if (config->dev_handle) {
        i2c_master_bus_rm_device(config->dev_handle);
        config->dev_handle = NULL;
    }
    
    if (config->bus_handle) {
        i2c_del_master_bus(config->bus_handle);
        config->bus_handle = NULL;
    }
    
    ESP_LOGI(TAG, "ICM20948 I2C deinitialized");
}