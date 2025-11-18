#include "esp_log.h"
#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_jpeg_common.h"
#include "esp_jpeg_dec.h"
#include "esp_jpeg_enc.h"
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
#include "freertos/semphr.h"
#include "esp_camera.h"
// SD , SPI
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "esp_camera.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

#include "Arduino.h" // TODO: do I need this?

#include "include/BLEModule.h"
#include "include/camera_pins.h"
#include "include/takeover_classification.hpp"
#include "include/surface_classification.hpp"
#include "include/led.hpp"
#include "include/distance.hpp"
#include "include/image_ring_buffer.hpp"

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

#define MOSAIC_WIDTH 16

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

    .pixel_format = PIXFORMAT_GRAYSCALE, // if grayscale: this line has to be changed to gray too: https://github.com/espressif/esp-dl/blob/f666d7671599dab74f67a3ecc2cb6668bf4e2de9/esp-dl/vision/image/dl_image_preprocessor.cpp#L22
    .frame_size = FRAMESIZE_QVGA,     // [<<320x240>> (QVGA, 4:3);FRAMESIZE_320X320, 240x176 (HQVGA, 15:11); 400x296 (CIF, 50:37)],FRAMESIZE_QVGA,FRAMESIZE_VGA

    // .jpeg_quality = 8, // 0-63 lower number means higher quality.  Reduce quality if stack overflow in cam_task
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
    ESP_LOGI("CAM", "Capturing picture...");
    camera_fb_t *pic = esp_camera_fb_get();
    if (!pic) {
        ESP_LOGE("CAM", "Failed to capture image");
        return false;
    }

    // Use pic->buf to access the image
    ESP_LOGI("CAM", "Picture taken! Height: %d, Width: %d, Len: %zu", pic->height, pic->width, pic->len);

    // // Create JPEG structure
    // dl::image::jpeg_img_t jpeg_img;
    // jpeg_img.data = (uint8_t*)malloc(pic->len);
    // if (jpeg_img.data) {
    //     memcpy(jpeg_img.data, pic->buf, pic->len);
    // } else {
    //     ESP_LOGE("CAM", "Failed to allocate memory for JPEG image");
    //     esp_camera_fb_return(pic);
    //     return false;
    // }
    // jpeg_img.data_len = pic->len;

    // Free previous image data (if any)
    if (output_img.data) {
        free(output_img.data);
        output_img.data = nullptr;
    }
    // Prepare output image metadata
    output_img.pix_type = camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    output_img.width = pic->width;
    output_img.height = pic->height;

    // Allocate and copy the frame buffer so the image remains valid
    // after we return the camera frame with esp_camera_fb_return(pic).
    size_t expected_size = dl::image::get_img_byte_size(output_img);
    size_t copy_size = MIN((size_t)pic->len, expected_size);
    output_img.data = malloc(expected_size);
    if (!output_img.data) {
        ESP_LOGE("CAM", "Failed to allocate memory for image copy (%zu bytes)", expected_size);
        esp_camera_fb_return(pic);
        return false;
    }
    // Zero the buffer if the camera provided less data than expected
    if (copy_size < expected_size) {
        memset(output_img.data, 0, expected_size);
    }
    memcpy(output_img.data, pic->buf, copy_size);

    esp_camera_fb_return(pic);
    // if(jpeg_img.data) {
    //     ESP_LOGI("CAM", "Free heap before freeing: %lu bytes", esp_get_free_heap_size());
    //     free(jpeg_img.data);
    //     jpeg_img.data = nullptr;
    // }
    return true;
}

// ---- Shared pipeline state ----
static dl::image::img_t g_buf[2];            // allocate these properly (global/static/heap)
static dl::image::img_t takeover_buf[2];     // allocate these properly (global/static/heap)
static dl::image::img_t surface_buf[2];      // allocate these properly (global/static/heap)
static volatile int g_read_idx = 0;          // which buffer consumers should read
static int g_write_idx = 1;                  // which buffer producer should write next
ImageRingBuffer ring_buffer;

// Optional: a mutex if your img_t needs guarded access during capture
// (generally not needed with strict read/write separation via indices)
static SemaphoreHandle_t g_buf_mutex; // (optional)

// ---- Capture -> fill buffer[write_idx] ----
static bool capture_into_buffer(int idx) {
    // capture_image should write into g_buf[idx]
    return capture_image(g_buf[idx]);
}

static bool convert_surface_into_buffer(int idx) {
    // Free previous data to prevent memory leak
    if (surface_buf[idx].data) {
        free(surface_buf[idx].data);
        surface_buf[idx].data = nullptr;
    }
    if (!convert_surface_image(&(g_buf[idx]), surface_buf[idx], camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888)) {
        ESP_LOGE("SURFACE", "Could not convert image");
        return false;
    }
    return true;
}

static bool convert_takeover_into_buffer(int idx) {
    dl::image::img_t converted_img;
    if (!convert_takeover_image(&(g_buf[idx]), converted_img, camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888)) {
        ESP_LOGE("TAKEOVER", "Could not convert image");
        return false;
    }

    // Add converted image to ring buffer
    if (!ring_buffer.add_image(converted_img)) {
        ESP_LOGE("TAKEOVER", "Could not add image to ring buffer");
        free(converted_img.data);
        return false;
    }
    
    // Free the original converted_img.data since it's been copied to ring buffer
    free(converted_img.data);
    converted_img.data = nullptr;
    
    ESP_LOGI("TAKEOVER", "Added image to ring buffer. Count: %d/%d", 
                ring_buffer.get_count(), RING_BUFFER_SIZE);

    // Check if the ring buffer is full
    if (!ring_buffer.is_full()) {
        ESP_LOGI("TAKEOVER", "Ring buffer has not been filled yet.");
        return false; // Not enough images yet
    } else {
        ESP_LOGI("TAKEOVER", "Ring buffer is full. Composing %d x %d image", 16/MOSAIC_WIDTH, MOSAIC_WIDTH);
        // Free previous data to prevent memory leak
        if (takeover_buf[idx].data) {
            free(takeover_buf[idx].data);
            takeover_buf[idx].data = nullptr;
        }
        // TODO: mutex
        if (!ring_buffer.compose_image(takeover_buf[idx], camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888, MOSAIC_WIDTH)) {
            ESP_LOGE("TAKEOVER", "Failed to compose %d x %d image", 16/MOSAIC_WIDTH, MOSAIC_WIDTH);
            return false;
        }
    }
    return true;
}

static void camera_capture_task(void *pvParameters) {
    TickType_t cLastWakeTime = xTaskGetTickCount();
    const TickType_t cFrequency = 150 / portTICK_PERIOD_MS; //delay for mS
    uint32_t last_capture_time = 0;
    // rolling buffer for camera capture FPS (kept outside loop so initialized once)
    static float capture_fps_buf[50] = {0};
    static int capture_fps_buf_idx = 0;
    static int capture_fps_buf_count = 0; // number of valid entries (<=50)
    for (;;) {
        // delay until maximum frequency
        cLastWakeTime = xTaskGetTickCount();
        vTaskDelayUntil( &cLastWakeTime, cFrequency );
        // calc and print framerate
        uint32_t now = xTaskGetTickCount();
        if (last_capture_time != 0) {
            float seconds = (now - last_capture_time) * portTICK_PERIOD_MS / 1000.0f;
            if (seconds > 0.0f) {
                float fps = 1.0f / seconds;

                // push into circular buffer
                capture_fps_buf[capture_fps_buf_idx] = fps;
                capture_fps_buf_idx = (capture_fps_buf_idx + 1) % 50;
                if (capture_fps_buf_count < 50) capture_fps_buf_count++;

                float sum = 0.0f;
                for (int i = 0; i < capture_fps_buf_count; ++i) sum += capture_fps_buf[i];
                float avg = sum / (float)capture_fps_buf_count;

                ESP_LOGI("CAM", "Capture framerate: %.2f FPS (avg last %d: %.2f FPS)", fps, capture_fps_buf_count, avg);
            }
        }
        last_capture_time = now;
        ESP_LOGI("CAM", "Free heap at start of loop: %lu bytes", esp_get_free_heap_size());

        // Capture into write buffer
        if (!capture_into_buffer(g_write_idx)) {
            ESP_LOGE("CAM", "capture failed");
            // If capture fails, still notify consumers? Usually no. Try again.
            continue;
        }

        if(!convert_takeover_into_buffer(g_write_idx)) {
            ESP_LOGE("CAM", "takeover conversion failed");
        }

        if(!convert_surface_into_buffer(g_write_idx)) {
            ESP_LOGE("CAM", "surface conversion failed");
        }

        // Publish the new frame by flipping read_idx atomically (single int write is atomic on ESP32)
        g_read_idx = g_write_idx;

        // Flip write index for the next capture
        g_write_idx ^= 1;
    }
}

static void surface_classification_task(void *pvParameters) {
    ESP_LOGI("SURFACE", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());
    uint32_t last_capture_time = 0;
    if (!initialize_surface_model()) {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("SURFACE", "Failed to initialize surface model");
        vTaskDelete(NULL);
    }
    // rolling buffer for surface classification FPS (kept outside loop so initialized once)
    static float surface_fps_buf[50] = {0};
    static int surface_fps_buf_idx = 0;
    static int surface_fps_buf_count = 0; // number of valid entries (<=16)
    // TickType_t sLastWakeTime = xTaskGetTickCount();
    // const TickType_t sFrequency = 50 / portTICK_PERIOD_MS; //delay for mS
    for (;;) {
        // delay until maximum frequency
        // sLastWakeTime = xTaskGetTickCount();
        // vTaskDelayUntil( &sLastWakeTime, sFrequency );
        // calc and print framerate
        uint32_t now = xTaskGetTickCount();
        if (last_capture_time != 0) {
            float seconds = (now - last_capture_time) * portTICK_PERIOD_MS / 1000.0f;
            if (seconds > 0.0f) {
                    float fps = 1.0f / seconds;
                    surface_fps_buf[surface_fps_buf_idx] = fps;
                    surface_fps_buf_idx = (surface_fps_buf_idx + 1) % 50;
                    if (surface_fps_buf_count < 50) surface_fps_buf_count++;

                    float sum = 0.0f;
                    for (int i = 0; i < surface_fps_buf_count; ++i) sum += surface_fps_buf[i];
                    float avg = sum / (float)surface_fps_buf_count;

                    ESP_LOGW("SURFACE", "Classification framerate: %.2f FPS (avg last %d: %.2f FPS)", fps, surface_fps_buf_count, avg);
            }
        }
        last_capture_time = now;

        int idx = g_read_idx;

        if (!process_surface_image(&surface_buf[idx])) {
            ESP_LOGE("SURFACE", "processing failed");
        }
    }
}

static void takeover_classification_task(void *pvParameters) {
    ESP_LOGI("TAKEOVER", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());
    uint32_t last_capture_time = 0;
    if (!initialize_takeover_model()) {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("TAKEOVER", "Failed to initialize takeover model");
        vTaskDelete(NULL);
    }
    // rolling buffer for takeover classification FPS (kept outside loop so initialized once)
    static float takeover_fps_buf[50] = {0};
    static int takeover_fps_buf_idx = 0;
    static int takeover_fps_buf_count = 0; // number of valid entries (<=16)
    // TickType_t tLastWakeTime = xTaskGetTickCount();
    // const TickType_t tFrequency = 50 / portTICK_PERIOD_MS; //delay for mS
    for (;;) {
        // delay until maximum frequency
        // tLastWakeTime = xTaskGetTickCount();
        // vTaskDelayUntil( &tLastWakeTime, tFrequency );
        // calc and print framerate
        uint32_t now = xTaskGetTickCount();
        if (last_capture_time != 0) {
            float seconds = (now - last_capture_time) * portTICK_PERIOD_MS / 1000.0f;
            if (seconds > 0.0f) {
                float fps = 1.0f / seconds;
                takeover_fps_buf[takeover_fps_buf_idx] = fps;
                takeover_fps_buf_idx = (takeover_fps_buf_idx + 1) % 50;
                if (takeover_fps_buf_count < 50) takeover_fps_buf_count++;

                float sum = 0.0f;
                for (int i = 0; i < takeover_fps_buf_count; ++i) sum += takeover_fps_buf[i];
                float avg = sum / (float)takeover_fps_buf_count;

                ESP_LOGW("TAKEOVER", "Classification framerate: %.2f FPS (avg last %d: %.2f FPS)", fps, takeover_fps_buf_count, avg);
            }
        }
        last_capture_time = now;

        int idx = g_read_idx;

        if (!process_takeover_image(&takeover_buf[idx])) {
            ESP_LOGE("TAKEOVER", "processing failed");
        }
    }
}

static void distance_task(void *pvParameters) {
    ESP_LOGI("DISTANCE", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());
    uint32_t last_distance_time = 0;
    if (!init_distance()) {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("DISTANCE", "Failed to initialize distance sensor"); 
        vTaskDelete(NULL);
    }
    TickType_t tLastWakeTime = xTaskGetTickCount();
    const TickType_t tFrequency = 400 / portTICK_PERIOD_MS; //delay for mS
    for (;;) {
        // delay until maximum frequency
        tLastWakeTime = xTaskGetTickCount();
        vTaskDelayUntil( &tLastWakeTime, tFrequency );
        // calc and print framerate
        uint32_t now = xTaskGetTickCount();
        if (last_distance_time != 0) {
            float seconds = (now - last_distance_time) * portTICK_PERIOD_MS / 1000.0f;
            if (seconds > 0.0f) {
                float fps = 1.0f / seconds;
                ESP_LOGI("DISTANCE", "Classification framerate: %.2f FPS", fps);
            }
        }
        last_distance_time = now;

        float distance = get_distance();
        ESP_LOGI("DISTANCE", "Distance: %.2f cm", distance);
    }
}

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

    set_LED(0, 255, 0, 10);

    // xTaskCreatePinnedToCore(distance_task,                  "distance", 8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(camera_capture_task,            "camera",   8192*2, NULL, 20, NULL, 1);
    vTaskDelay(3500 / portTICK_PERIOD_MS);
    // xTaskCreatePinnedToCore(surface_classification_task,   "surface", 8192*2, NULL, 19, NULL, 0);
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    xTaskCreatePinnedToCore(takeover_classification_task,    "takeover",  8192*2, NULL, 19, NULL, 1);
    }
