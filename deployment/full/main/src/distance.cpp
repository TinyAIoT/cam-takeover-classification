#include "distance.hpp"
#include "BLEModule.h"

static VL53L8CX sensor_vl53l8cx(&Wire, -1, -1);
float oldVl53l8cxMin = -1.0;

const bool init_distance() {
    Wire.begin(PIN_QWIIC_SDA,PIN_QWIIC_SCL);

    ESP_LOGI("DISTANCE", "start init");
    Wire.begin(PIN_QWIIC_SDA,PIN_QWIIC_SCL);
    vTaskDelay(500 / portTICK_PERIOD_MS);
    int ret = sensor_vl53l8cx.set_i2c_address(0x51); // need to change address, because default address is shared with other sensor
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to set I2C address");
        return false;
    }

    Wire.setClock(1000000); // vl53l8cx can operate at 1MHz
    ret = sensor_vl53l8cx.begin();
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to begin sensor");
        return false;
    }
    ret = sensor_vl53l8cx.init();
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to init sensor");
        return false;
    }
    ret = sensor_vl53l8cx.set_ranging_frequency_hz(30);
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to set ranging frequency");
        return false;
    }
    ret = sensor_vl53l8cx.set_resolution(VL53L8CX_RESOLUTION_8X8);
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to set resolution");
        return false;
    }
    ret = sensor_vl53l8cx.start_ranging();
    if (ret != 0) {
        ESP_LOGE("DISTANCE", "Failed to start ranging");
        return false;
    }
    Wire.setClock(100000); // lower the I2C clock to 0.1MHz again for compatibility with other sensors

    ESP_LOGI("DISTANCE", "init done");
    return true;
}

const float get_distance() {
    VL53L8CX_ResultsData Results;
    uint8_t NewDataReady = 0;
    uint8_t status;

    Wire.setClock(1000000); // vl53l8cx can operate at 1MHz
    status = sensor_vl53l8cx.check_data_ready(&NewDataReady);

    if ((!status) && (NewDataReady != 0)) {
        sensor_vl53l8cx.get_ranging_data(&Results);
        Wire.setClock(100000); // lower the I2C clock to 0.1MHz again for compatibility with other sensors
        float minStatus5 = 10000.0; // bigger than the maximum measurable distance
        float minStatus69 = 10000.0; // bigger than the maximum measurable distance
        for(int i = 0; i < VL53L8CX_RESOLUTION_8X8*VL53L8CX_NB_TARGET_PER_ZONE; i++) {
            float distance = ((&Results)->distance_mm[i])/10;
            float target_status = (&Results)->target_status[i];
            if(target_status == 5 && minStatus5 > distance) {
                minStatus5 = distance;
            } else if((target_status == 6 || target_status == 9) && minStatus69 > distance) {
                minStatus69 = distance;
            }
        }
        if (minStatus5 < 10000.0 && minStatus5 >=0) {
            oldVl53l8cxMin = minStatus5;
        } else if (minStatus69 < 10000.0 && minStatus69 >=0) {
            oldVl53l8cxMin = minStatus69;
        } else {
            oldVl53l8cxMin = 0.0;
        }
    }
    uint8_t distance[1] = {
        (uint8_t)oldVl53l8cxMin
    };
    notify_distance(distance);

    return oldVl53l8cxMin;
}