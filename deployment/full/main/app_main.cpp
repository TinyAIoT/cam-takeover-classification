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
    
    // Allocate memory for the image data
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

extern "C" void app_main(void) {
    if (ESP_OK != init_camera())
    {
        ESP_LOGE("SD", "Camera initialization failed. Look into init_camera()");
        return;
    }

    while (true)
    {
        ESP_LOGI("MEM", "Free heap at start of loop: %lu bytes", esp_get_free_heap_size());
        dl::image::img_t img;
        if (!capture_image(img)) {
            ESP_LOGE("img", "Could not take picture");
            return;
        }

        float takeover_score = 0.0f;
        const char* takeover_category = nullptr;
        if (process_takeover_image(img, takeover_score, &takeover_category)) {
            ESP_LOGI("TAKEOVER", "Takeover Score: %.2f, Category: %s", takeover_score, takeover_category);
        }

        float surface_score = 0.0f;
        const char* surface_category = nullptr;
        if (process_surface_image(img, surface_score, &surface_category)) {
            ESP_LOGI("SURFACE", "Surface Score: %.2f, Category: %s", surface_score, surface_category);
        }   

        // Free the original camera image data
        if (img.data) {
            free(img.data);
            img.data = nullptr;
        }
        
    }
}
