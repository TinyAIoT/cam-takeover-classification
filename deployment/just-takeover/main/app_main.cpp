#include "esp_log.h"
#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
#include "dl_image_jpeg.hpp"
#include "bsp/esp-bsp.h"
#include <esp_system.h>
#include <nvs_flash.h>
#include <string.h>
#include <sys/param.h>
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "host/ble_hs.h"

#include "esp_jpeg_enc.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
// SD , SPI
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "esp_camera.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cstring>



#include "include/BLEModule.h"
#include "include/camera_pins.h"
#include "include/takeover_classification.hpp"
#include "include/led.hpp"
#include "include/distance.hpp"

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

static const char *device_name = "senseBox:bike[XXX]";

// Camera Module pin mapping
static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,

    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 20000000, // XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_RGB565, // PIXFORMAT_RGB565 , PIXFORMAT_JPEG
    .frame_size = FRAMESIZE_QVGA,     // [<<320x240>> (QVGA, 4:3);FRAMESIZE_320X320, 240x176 (HQVGA, 15:11); 400x296 (CIF, 50:37)],FRAMESIZE_QVGA,FRAMESIZE_VGA

    .jpeg_quality = 8, // 0-63 lower number means higher quality.  Reduce quality if stack overflow in cam_task
    .fb_count = 2,     // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0 // optional
};

static esp_err_t init_camera(void) {
    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE("CAM", "Camera Init Failed");
    }
    return err;
}

static bool capture_image(dl::image::img_t &output_img) {
    ESP_LOGI("CAM", "Taking picture...");
    camera_fb_t *pic = esp_camera_fb_get();
    if (!pic) {
        ESP_LOGE("CAM", "Failed to capture image");
        return false;
    }

    // Use pic->buf to access the image
    ESP_LOGI("CAM", "Picture taken! Its size was: %zu bytes", pic->len);
    ESP_LOGW("image_dim", "Height: %d, Width: %d, Len: %zu", pic->height, pic->width, pic->len);

    // Allocate memory and copy the image data before returning the frame buffer
    output_img.height = pic->height;
    output_img.width = pic->width;
    output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;

    if (output_img.data) {
        free(output_img.data);
        output_img.data = nullptr;
    }
    output_img.data = malloc(pic->len);

    if (!output_img.data) {
        ESP_LOGE("CAM", "Failed to allocate memory for image data");
        esp_camera_fb_return(pic);
        return false;
    }
    
    // Copy the image data
    memcpy(output_img.data, pic->buf, pic->len);

    esp_camera_fb_return(pic);
    return true;
}

// ---- Shared pipeline state ----
static dl::image::img_t g_buf[2];            // allocate these properly (global/static/heap)
static volatile int g_read_idx = 0;          // which buffer consumers should read
static int g_write_idx = 1;                  // which buffer producer should write next


// Optional: a mutex if your img_t needs guarded access during capture
// (generally not needed with strict read/write separation via indices)
static SemaphoreHandle_t g_buf_mutex; // (optional)

// ---- Capture -> fill buffer[write_idx] ----
static bool capture_into_buffer(int idx) {
    // capture_image should write into g_buf[idx]
    return capture_image(g_buf[idx]);
}

SemaphoreHandle_t vl53l8cxInitDone;

extern "C" void app_main(void) {
    init_LED(); // initialize RGB-LED
    set_LED(255, 255, 0, 20);

    esp_err_t ret = nimble_port_init();
    if (ret != ESP_OK) {
        MODLOG_DFLT(ERROR, "Failed to init nimble %d \n", ret);
        set_LED(255, 50, 0, 20);
        return;
    }
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.reset_cb = on_reset;
    int rc = gatt_svr_init();
    assert(rc == 0);
    rc = ble_svc_gap_device_name_set(device_name);
    assert(rc == 0);

    nimble_port_freertos_init(host_task);

    if (ESP_OK != init_camera()) {
        ESP_LOGE("CAM", "Camera init failed");
        set_LED(255, 50, 0, 20);
        return;
    }

    if (!initialize_takeover_model()) {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("TAKEOVER", "Failed to initialize takeover model");
        return;
    }

    vl53l8cxInitDone = xSemaphoreCreateBinary();

    // for some reason the vl53l8cx init needs to run specifically on core 1...
    xTaskCreatePinnedToCore([](void*){
        if (!init_distance()) {
            set_LED(255, 0, 150, 20);
            ESP_LOGE("DISTANCE", "Failed to initialize distance sensor"); 
        } else {
            xSemaphoreGive(vl53l8cxInitDone);
        }
        // handle ok/!ok
        vTaskDelete(nullptr);
    }, "sensorTask", 12288, nullptr, 1, nullptr, 1); // often core 1

    // Wait for vl53l8cx init to finish
    if (xSemaphoreTake(vl53l8cxInitDone, portMAX_DELAY) == pdTRUE) {
        printf("Sensor init complete!\n");
    }

    set_LED(0, 255, 0, 10);

    uint32_t last_capture_time = 0;
    for (;;) {
        ESP_LOGI("CAM", "Free heap at start of loop: %lu bytes", esp_get_free_heap_size());

        // Capture into write buffer
        if (!capture_into_buffer(g_write_idx)) {
            ESP_LOGE("CAM", "capture failed");
            // If capture fails, still notify consumers? Usually no. Try again.
            continue;
        }

        // Framerate calculation
        uint32_t now = xTaskGetTickCount();
        if (last_capture_time != 0) {
            float seconds = (now - last_capture_time) * portTICK_PERIOD_MS / 1000.0f;
            if (seconds > 0.0f) {
                float fps = 1.0f / seconds;
                ESP_LOGI("CAM", "Capture framerate: %.2f FPS", fps);
            }
        }
        last_capture_time = now;

        // Publish the new frame by flipping read_idx atomically (single int write is atomic on ESP32)
        g_read_idx = g_write_idx;

        // Flip write index for the next capture
        g_write_idx ^= 1;

        int idx = g_read_idx;

        if (!process_takeover_image(&g_buf[idx])) {
            ESP_LOGW("TAKEOVER", "processing failed");
        }

        float distance = get_distance();
        ESP_LOGI("DISTANCE", "Distance: %.2f cm", distance);
    }
}
