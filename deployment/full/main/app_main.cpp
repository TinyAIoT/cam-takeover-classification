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

#include "include/takeover_classification.hpp"
#include "include/surface_classification.hpp"

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

#include "camera_pins.h"

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

    // Free previously allocated memory to avoid leaks and buffer overflows

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

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// ---- Shared pipeline state ----
static dl::image::img_t g_buf[2];            // allocate these properly (global/static/heap)
static volatile int g_read_idx = 0;          // which buffer consumers should read
static int g_write_idx = 1;                  // which buffer producer should write next

// event to trigger surface and takeover classification
static EventGroupHandle_t g_frame_evt;

// Semaphores to signal producer that each classifier finished
static SemaphoreHandle_t g_sem_done_surface;
static SemaphoreHandle_t g_sem_done_takeover;

// Optional: a mutex if your img_t needs guarded access during capture
// (generally not needed with strict read/write separation via indices)
static SemaphoreHandle_t g_buf_mutex; // (optional)

// ---- Capture -> fill buffer[write_idx] ----
static bool capture_into_buffer(int idx) {
    // capture_image should write into g_buf[idx]
    return capture_image(g_buf[idx]);
}

static void camera_capture_task(void *pvParameters) {
    uint32_t last_capture_time = 0;
    for (;;) {
        ESP_LOGI("CAM", "Free heap at start of loop: %lu bytes", esp_get_free_heap_size());

        // Wait until both consumers are done with the previous frame
        xSemaphoreTake(g_sem_done_surface, portMAX_DELAY);
        xSemaphoreTake(g_sem_done_takeover, portMAX_DELAY);

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

        xEventGroupSetBits(g_frame_evt, BIT0 | BIT1);  // release both Surface and takeover
    }
}

static void surface_classification_task(void *pvParameters) {
    if (!initialize_surface_model()) {
        ESP_LOGE("SURFACE", "Failed to initialize surface model");
        vTaskDelete(NULL);
    }
    for (;;) {
        // Wait for a new frame
        xEventGroupWaitBits(g_frame_evt, BIT0, pdTRUE, pdTRUE, portMAX_DELAY); // SURFACE waits BIT0 (auto-clear with pdTRUE)
        
        ESP_LOGI("SURFACE", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());

        // Read current frame index AFTER take: memory order is fine through semaphore
        int idx = g_read_idx;

        float score = 0.0f;
        const char* category = NULL;
        if (process_surface_image(&g_buf[idx], score, &category)) {
            ESP_LOGI("SURFACE", "Score: %.2f, Category: %s", score, category ? category : "NULL");
        } else {
            ESP_LOGW("SURFACE", "processing failed");
        }

        xSemaphoreGive(g_sem_done_surface);
    }
}

static void takeover_classification_task(void *pvParameters) {
    if (!initialize_takeover_model()) {
        ESP_LOGE("TAKEOVER", "Failed to initialize takeover model");
        vTaskDelete(NULL);
    }
    for (;;) {
        // Wait for a new frame
        xEventGroupWaitBits(g_frame_evt, BIT1, pdTRUE, pdTRUE, portMAX_DELAY); // TAKEOVER waits BIT1

        ESP_LOGI("TAKEOVER", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());

        int idx = g_read_idx;

        float score = 0.0f;
        const char* category = NULL;
        if (process_takeover_image(&g_buf[idx], score, &category)) {
            ESP_LOGI("TAKEOVER", "Score: %.2f, Category: %s", score, category ? category : "NULL");
        } else {
            ESP_LOGW("TAKEOVER", "processing failed");
        }

        xSemaphoreGive(g_sem_done_takeover);
    }
}

extern "C" void app_main(void) {
    if (ESP_OK != init_camera()) {
        ESP_LOGE("CAM", "Camera init failed");
        return;
    }

    // Init semaphores
    g_frame_evt = xEventGroupCreate();
    configASSERT(g_frame_evt != NULL);
    g_sem_done_surface   = xSemaphoreCreateBinary();
    g_sem_done_takeover  = xSemaphoreCreateBinary();

    // On the very first iteration, the producer will block waiting for both "done".
    // Give them once so the first capture can happen immediately:
    xSemaphoreGive(g_sem_done_surface);
    xSemaphoreGive(g_sem_done_takeover);

    xTaskCreatePinnedToCore(camera_capture_task,       "camera",   8192*2, NULL, 12, NULL, 0);
    xTaskCreatePinnedToCore(surface_classification_task,"surface",  8192*2, NULL, 17, NULL, 0);
    xTaskCreatePinnedToCore(takeover_classification_task,"takeover",8192*2, NULL, 20, NULL, 1);
}
