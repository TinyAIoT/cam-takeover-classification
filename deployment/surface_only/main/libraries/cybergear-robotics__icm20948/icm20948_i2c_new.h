#ifndef _ICM_20948_I2C_H_
#define _ICM_20948_I2C_H_

#include "driver/i2c_master.h"
#include "icm20948.h"

typedef struct
{
    i2c_port_t i2c_port;
    uint8_t i2c_addr;
    uint32_t scl_speed_hz;
    i2c_master_bus_handle_t bus_handle;
    i2c_master_dev_handle_t dev_handle;
} icm20948_config_i2c_t;

// Initialize ICM20948 with new I2C driver
icm20948_status_e icm20948_init_i2c_new(icm20948_device_t *device, icm20948_config_i2c_t *config);

// Cleanup I2C resources
void icm20948_deinit_i2c_new(icm20948_config_i2c_t *config);

// Internal I2C functions using new driver
icm20948_status_e icm20948_internal_write_i2c_new(uint8_t reg, uint8_t *data, uint32_t len, void *user);
icm20948_status_e icm20948_internal_read_i2c_new(uint8_t reg, uint8_t *buff, uint32_t len, void *user);

#endif