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
#include <esp_timer.h>
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


#include "include/BLEModule.h"
#include "include/camera_pins.h"
#include "include/surface_classification.hpp"
#include "include/led.hpp"
#include "include/fps_tracker.hpp"

#include "include/imu.h"

// #include "include/distance.hpp"
// #include "include/image_ring_buffer.hpp"

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

static FPSTracker camera_fps("CAM_FPS");
static FPSTracker classification_fps("CLASSIFY_FPS");
static FPSTracker imu_fps("IMU_FPS");

static const char *device_name = "senseBox:bike[XXX]";

// ---- Pipeline State ----
#define MIN_IMU_READINGS 20

// Image buffers for double-buffering
// static dl::image::img_t g_buf[2];         // allocate these properly (global/static/heap)
static dl::image::img_t surface_buf[2];      // allocate these properly (global/static/heap)
// static volatile int g_read_idx = 0;          // which buffer consumers should read
// static int g_write_idx = 1;                  // which buffer producer should write next
static volatile int surface_read_idx = 0;
static int surface_write_idx = 1;
static portMUX_TYPE surface_buf_spinlock = portMUX_INITIALIZER_UNLOCKED;

// IMU data collection
static imu_data_t imu_buffer[MIN_IMU_READINGS];
static volatile int imu_count = 0;
static SemaphoreHandle_t imu_ready_sem;

// Optional: a mutex if your img_t needs guarded access during capture
// (generally not needed with strict read/write separation via indices)
// static SemaphoreHandle_t g_buf_mutex; // (optional)
// Handle for the classification task so the camera task can notify it

static TaskHandle_t sensor_task_handle = NULL;
static TaskHandle_t classification_task_handle = NULL;

// State flags
static volatile bool classification_ready = false;

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

    .pixel_format = PIXFORMAT_RGB565, // if grayscale: this line has to be changed to gray too: https://github.com/espressif/esp-dl/blob/f666d7671599dab74f67a3ecc2cb6668bf4e2de9/esp-dl/vision/image/dl_image_preprocessor.cpp#L22
    .frame_size = FRAMESIZE_QVGA,     // [<<320x240>> (QVGA, 4:3);FRAMESIZE_320X320, 240x176 (HQVGA, 15:11); 400x296 (CIF, 50:37)],FRAMESIZE_QVGA,FRAMESIZE_VGA

    // .jpeg_quality = 8, // 0-63 lower number means higher quality.  Reduce quality if stack overflow in cam_task
    .fb_count = 1,     // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0 // optional
};

// Helper function for surface image conversion from camera frame buffer
bool convert_surface_image_from_fb(camera_fb_t *fb, dl::image::img_t &output_img) {
    // Create temporary image from frame buffer
    dl::image::img_t temp_img = {
        .data = fb->buf,
        .width = (uint16_t)fb->width,
        .height = (uint16_t)fb->height,
        .pix_type = (camera_config.pixel_format == PIXFORMAT_GRAYSCALE) ? 
                    dl::image::DL_IMAGE_PIX_TYPE_GRAY : 
                    dl::image::DL_IMAGE_PIX_TYPE_RGB565
    };
    
    // Convert using your existing function
    return convert_surface_image(&temp_img, output_img, 
                                camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? 
                                dl::image::DL_IMAGE_PIX_TYPE_GRAY : 
                                dl::image::DL_IMAGE_PIX_TYPE_RGB888);
}

static esp_err_t init_camera(void) {
    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE("CAM", "Camera Init Failed");
    }
    return err;
}

static bool capture_and_process_image(dl::image::img_t &output_img) {
    ESP_LOGI("CAM", "Capturing picture...");
    camera_fb_t *pic = esp_camera_fb_get();
    if (!pic) {
        ESP_LOGE("CAM", "Failed to capture image");
        return false;
    }
    
    // Validate frame buffer
    if (!pic->buf || pic->len == 0) {
        ESP_LOGE("CAM", "Invalid frame buffer: buf=%p, len=%zu", pic->buf, pic->len);
        esp_camera_fb_return(pic);
        return false;
    }

    // Use pic->buf to access the image
    ESP_LOGI("CAM", "Picture taken! Height: %d, Width: %d, Len: %zu", pic->height, pic->width, pic->len);

    // Free previous image data (if any)
    if (output_img.data) {
        free(output_img.data);
        output_img.data = nullptr;
    }

   
    // Convert and crop for surface classification
    if (!convert_surface_image_from_fb(pic, output_img)) {
        ESP_LOGE("CAM", "Surface conversion failed");
        esp_camera_fb_return(pic);
        return false;
    }

    esp_camera_fb_return(pic);
    return true;
}

    // // Prepare output image metadata
    // output_img.pix_type = camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    // output_img.width = pic->width;
    // output_img.height = pic->height;

    // // Allocate and copy the frame buffer so the image remains valid
    // // after we return the camera frame with esp_camera_fb_return(pic).
    // size_t expected_size = dl::image::get_img_byte_size(output_img);
    // if (expected_size == 0) {
    //     ESP_LOGE("CAM", "Calculated image size is zero");
    //     esp_camera_fb_return(pic);
    //     return false;
    // }

    // // Try to allocate in PSRAM first for large buffers
    // output_img.data = heap_caps_malloc(expected_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    // if (!output_img.data) {
    //     // Fall back to regular heap
    //     output_img.data = malloc(expected_size);
    // }
    
    // if (!output_img.data) {
    //     ESP_LOGE("CAM", "Failed to allocate memory for image copy (%zu bytes)", expected_size);
    //     ESP_LOGE("CAM", "Free heap: %lu bytes, Free PSRAM: %lu bytes", 
    //              heap_caps_get_free_size(MALLOC_CAP_8BIT), 
    //              heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    //     esp_camera_fb_return(pic);
    //     return false;
    // }

    // size_t copy_size = MIN((size_t)pic->len, expected_size);
    // // Zero the buffer if the camera provided less data than expected
    // if (copy_size < expected_size) {
    //     ESP_LOGW("CAM", "Camera data (%zu) < expected (%zu), padding with zeros", copy_size, expected_size);
    //     memset(output_img.data, 0, expected_size);
    // }
    // memcpy(output_img.data, pic->buf, copy_size);

    // esp_camera_fb_return(pic);
    // // if(jpeg_img.data) {
    // //     ESP_LOGI("CAM", "Free heap before freeing: %lu bytes", esp_get_free_heap_size());
    // //     free(jpeg_img.data);
    // //     jpeg_img.data = nullptr;
    // // }
    // return true;
// }

// ...existing code...

// // Add proper cleanup for global buffers
// static void cleanup_buffers() {
//     for (int i = 0; i < 2; i++) {
//         if (g_buf[i].data) {
//             free(g_buf[i].data);
//             g_buf[i].data = nullptr;
//         }
//         if (surface_buf[i].data) {
//             free(surface_buf[i].data);
//             surface_buf[i].data = nullptr;
//         }
//     }
// }

// Initialize buffers properly
static void init_buffers() {
    for (int i = 0; i < 2; i++) {
        // memset(&g_buf[i], 0, sizeof(dl::image::img_t));
        memset(&surface_buf[i], 0, sizeof(dl::image::img_t));
    }
}


// ---- Capture -> fill buffer[write_idx] ----
// static bool capture_into_buffer(int idx) {
//     // capture_image should write into g_buf[idx]
//     return capture_image(g_buf[idx]);
// }

// static bool convert_surface_into_buffer(int idx) {
//     // Free previous data to prevent memory leak
//     if (surface_buf[idx].data) {
//         free(surface_buf[idx].data);
//         surface_buf[idx].data = nullptr;
//     }
//     if (!convert_surface_image(&(g_buf[idx]), surface_buf[idx], camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888)) {
//         ESP_LOGE("SURFACE", "Could not convert image");
//         return false;
//     }
//     return true;
// }


// // Deep-copy helper: copy image metadata and bytes from src into dst.
// static bool copy_image(const dl::image::img_t &src, dl::image::img_t &dst) {
//     // Free destination if previously allocated
//     if (dst.data) {
//         free(dst.data);
//         dst.data = nullptr;
//     }

//     dst.pix_type = src.pix_type;
//     dst.width = src.width;
//     dst.height = src.height;

//     size_t sz = dl::image::get_img_byte_size(src);
//     if (sz == 0) {
//         ESP_LOGE("IMG", "Source image has zero size");
//         return false;
//     }

//     dst.data = malloc(sz);
//     if (!dst.data) {
//         ESP_LOGE("IMG", "Failed to allocate %zu bytes for image copy", sz);
//         return false;
//     }

//     memcpy(dst.data, src.data, sz);
//     return true;
// }

// Add proper synchronization
// static volatile bool g_classification_ready = false;
// static portMUX_TYPE g_buf_spinlock = portMUX_INITIALIZER_UNLOCKED;

// Add these timing variables at the top of the file
static struct {
    uint64_t camera_capture_us;
    uint64_t image_processing_us;
    uint64_t buffer_swap_us;
    uint64_t notification_us;
    uint64_t total_imu_inactive_us;
} timing_stats = {0};

static void sensor_task(void *pvParameters) {
    ESP_LOGI("SENSOR", "Sensor task started on core %d", xPortGetCoreID());
    // TickType_t cLastWakeTime = xTaskGetTickCount();
    // const TickType_t cFrequency = pdMS_TO_TICKS(200); // 143 ms
    // TickType_t last_classification_check = xTaskGetTickCount();
    // int camera_iteration = 0;

    // Initialize IMU
    if (imu_init() != ESP_OK) {
        ESP_LOGE("SENSOR", "Failed to initialize IMU");
        set_LED(255, 0, 0, 50);
        vTaskDelete(NULL);
        return;
    }

    imu_data_t imu_data;
    int cycle_count = 0;

    for (;;) {
        // Phase 1: Collect IMU readings
        ESP_LOGI("SENSOR", "=== Cycle %d: Collecting IMU readings ===", ++cycle_count);
        
        uint64_t imu_collection_start = esp_timer_get_time();

        imu_count = 0;
        for (int i = 0; i < MIN_IMU_READINGS; i++) {
            if (imu_read(&imu_data) == ESP_OK && imu_data.valid) {
                memcpy(&imu_buffer[i], &imu_data, sizeof(imu_data_t));
                imu_count++;
                imu_fps.update();
                
                ESP_LOGD("SENSOR", "IMU %d/%d: Acc[%d,%d,%d] Gyr[%d,%d,%d]", 
                         i+1, MIN_IMU_READINGS,
                         imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
                         imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z);
            } else {
                ESP_LOGW("SENSOR", "Failed to read IMU data %d/%d", i+1, MIN_IMU_READINGS);
                i--; // Retry this reading
            }
            
            vTaskDelay(pdMS_TO_TICKS(10)); // 100Hz IMU sampling
        }
        
        
        uint64_t imu_collection_end = esp_timer_get_time();
        uint64_t imu_collection_time = imu_collection_end - imu_collection_start;
        
        ESP_LOGI("SENSOR", "Collected %d IMU readings in %llu ms", 
                 imu_count, imu_collection_time / 1000);
        
        
        ESP_LOGI("SENSOR", "=== IMU INACTIVE START - Camera capture phase ===");
        
        // Sub-phase 2a: Camera capture
        uint64_t camera_start = esp_timer_get_time();
        
        bool camera_success = capture_and_process_image(surface_buf[surface_write_idx]);
        
        uint64_t camera_end = esp_timer_get_time();
        uint64_t cam_imu_inactive_start = esp_timer_get_time();

        timing_stats.camera_capture_us = camera_end - camera_start;
        
        if (!camera_success) {
            ESP_LOGE("SENSOR", "Camera capture failed after %llu ms, retrying...", 
                     timing_stats.camera_capture_us / 1000);
            continue;
        }
        
        camera_fps.update();

        ESP_LOGI("SENSOR", "Camera capture completed in %llu ms", 
            timing_stats.camera_capture_us / 1000);
        
        

        
        // Phase 3: Publish image for classification (atomic swap)
        portENTER_CRITICAL(&surface_buf_spinlock);
        surface_read_idx = surface_write_idx;
        surface_write_idx ^= 1; // Flip between 0 and 1
        portEXIT_CRITICAL(&surface_buf_spinlock);
        
        // Phase 4: Signal classification task and continue IMU collection
        if (classification_task_handle && classification_ready) {
            ESP_LOGI("SENSOR", "=== Phase 3: Notifying classification task ===");
            BaseType_t result = xTaskNotifyGive(classification_task_handle);
            if (result != pdPASS) {
                ESP_LOGE("SENSOR", "Failed to notify classification task!");
            }
        } else {
            ESP_LOGW("SENSOR", "Classification not ready: handle=%p, ready=%d", 
                     classification_task_handle, classification_ready);
        }
        
        // Small delay before next cycle
        vTaskDelay(pdMS_TO_TICKS(10));

        uint64_t cam_imu_inactive_end = esp_timer_get_time();
        timing_stats.total_imu_inactive_us = cam_imu_inactive_end - cam_imu_inactive_start;

        ESP_LOGI("SENSOR", "=== IMU + CAM INACTIVE END - Total inactive time including swap and delay of 10 ms: %llu ms ===", 
                 timing_stats.total_imu_inactive_us / 1000);
    }


    // for (;;) {
    //     // delay until maximum frequency (keep `cLastWakeTime` across iterations)
    //     ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    //     vTaskDelayUntil(&cLastWakeTime, cFrequency);
        
    //     camera_fps.update();
    //     camera_iteration++;
    

    //     ESP_LOGI("CAM", "Free heap at start of loop: %lu bytes", esp_get_free_heap_size());

    //     // Capture into write buffer
    //     if (!capture_into_buffer(g_write_idx)) {
    //         ESP_LOGE("CAM", "capture failed");
    //         // If capture fails, still notify consumers? Usually no. Try again.
    //         continue;
    //     }

    //     if(!convert_surface_into_buffer(g_write_idx)) {
    //         ESP_LOGE("CAM", "surface conversion failed");
    //         continue;
    //     }

    //     // Thread-safe buffer swap
    //     portENTER_CRITICAL(&g_buf_spinlock);
    //     // Publish the new frame by flipping read_idx atomically (single int write is atomic on ESP32)
    //     g_read_idx = g_write_idx;
    //     // Flip write index for the next capture
    //     g_write_idx ^= 1;
    //     portEXIT_CRITICAL(&g_buf_spinlock);

    //     // Notify classification task that a new frame is available
    //     if (classification_task_handle != NULL && g_classification_ready) {
    //         ESP_LOGW("CAM", "Sending notification to classification task (iter %d)", camera_iteration);
    //         BaseType_t notify_result = xTaskNotifyGive(classification_task_handle);
    //         if (notify_result != pdPASS) {
    //             ESP_LOGE("CAM", "Failed to notify classification task!");
    //         } else {
    //             ESP_LOGW("CAM", "Notification sent successfully");
    //         }
    //     } else {
    //         ESP_LOGW("CAM", "Classification not ready: handle=%p, ready=%d", 
    //                  classification_task_handle, g_classification_ready);
    //     }

    // }
}

// // In your main function or a dedicated IMU task
// void imu_task(void *pvParameters) {
//     // Initialize IMU
//     if (imu_init() != ESP_OK) {
//         ESP_LOGE("IMU_TASK", "Failed to initialize IMU");
//         vTaskDelete(NULL);
//         return;
//     }
    
//     imu_data_t imu_data;
//     TickType_t imuLastWakeTime = xTaskGetTickCount();
//     const TickType_t imuFrequency = pdMS_TO_TICKS(10); // 10 ms
//     int imu_iteration = 0;
//     for (;;) {
//         vTaskDelayUntil(&imuLastWakeTime, imuFrequency);
//         imu_fps.update();

//         if (imu_read(&imu_data) == ESP_OK && imu_data.valid) {
//             ESP_LOGW("IMU", "Acc[%d,%d,%d] Gyr[%d,%d,%d] Temp[%d]", 
//                      imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
//                      imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z,
//                      imu_data.temp);
//             imu_iteration++;
//         }
        
//         // capture at least 10 IMU reads before notifying camera task, to coalesce updates and avoid overwhelming the camera task if it's busy
//         // Notify camera task that new IMU data is available (if needed for fusion)

//         if (camera_capture_task_handle != NULL && imu_iteration >= 10) {
//             ESP_LOGW("IMU", "Sending notification to camera capture task (iter %d)", imu_iteration);
//             BaseType_t notify_result = xTaskNotifyGive(camera_capture_task_handle);
//             if (notify_result != pdPASS) {
//                 ESP_LOGE("IMU", "Failed to notify camera capture task!");
//             } else {
//                 ESP_LOGW("IMU", "Notification sent successfully");
//             }
//             imu_iteration = 0; // reset iteration count after notification
//         }

//     }
// }

static void classification_task(void *pvParameters) {
    ESP_LOGI("CLASSIFY", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());

    if (!initialize_surface_model()) {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("SURFACE", "Failed to initialize surface model");
        vTaskDelete(NULL);
        return;
    }
    
    // Signal that classification is ready
    classification_ready = true;
    ESP_LOGI("CLASSIFY", "Surface model initialized, ready for classification");

    int iteration_count = 0;

    for (;;) {
        // Wait for notification from camera task that a new frame is available
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    
        iteration_count++;
        classification_fps.update();

        ESP_LOGI("CLASSIFY", "=== Classification iteration %d ===", iteration_count);

        // Get current buffer index atomically
        int idx;
        portENTER_CRITICAL(&surface_buf_spinlock);
        idx = surface_read_idx;
        portEXIT_CRITICAL(&surface_buf_spinlock);
             
        
        // Validate buffer before processing
        if (!surface_buf[idx].data) {
            ESP_LOGW("CLASSIFY", "No valid data in buffer %d", idx);
            continue;
        }

        // Process surface classification
        TickType_t start_time = xTaskGetTickCount();
        bool success = process_surface_image(&surface_buf[idx]);
        TickType_t end_time = xTaskGetTickCount();
        
        uint32_t elapsed_ms = (end_time - start_time) * portTICK_PERIOD_MS;
        
        if (!success) {
            ESP_LOGE("CLASSIFY", "Surface processing failed (took %lu ms)", elapsed_ms);
            set_LED(255, 0, 0, 30); // Red - error
        } else {
            ESP_LOGI("CLASSIFY", "Surface processing succeeded (took %lu ms)", elapsed_ms);
            set_LED(0, 255, 0, 10); // Green - success
        }
        
        // Log performance periodically
        if (iteration_count % 5 == 0) {
            ESP_LOGI("CLASSIFY", "Performance - Cam: %.1f fps, IMU: %.1f fps, Classify: %.1f fps",
                     camera_fps.get_current(), imu_fps.get_current(), classification_fps.get_current());
        }
    }
}




extern "C" void app_main(void) {
    // Initialize LED and show startup
    init_LED();
    set_LED(255, 255, 0, 20);

    init_buffers();

    // Create semaphores
    imu_ready_sem = xSemaphoreCreateBinary();
    if (imu_ready_sem == NULL) {
        ESP_LOGE("MAIN", "Failed to create IMU ready semaphore");
        set_LED(255, 0, 0, 50); // Red - fatal error
        return;
    }

    // Only show warnings and errors at runtime (suppress INFO/DEBUG)
// Set logging levels
    esp_log_level_set("*", ESP_LOG_INFO);
    esp_log_level_set("SENSOR", ESP_LOG_DEBUG);
    esp_log_level_set("CLASSIFY", ESP_LOG_DEBUG);
    
    // Initialize NVS (if not already done)
    esp_err_t nvs_ret = nvs_flash_init();
    if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES || nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_ret);


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

    // Create tasks
    ESP_LOGI("MAIN", "Creating tasks...");

    // Core 0: Sensor task (IMU + Camera)
    BaseType_t sensor_result = xTaskCreatePinnedToCore(
        sensor_task, "sensor", 12288, NULL, 15, &sensor_task_handle, 0);
    if (sensor_result != pdPASS) {
        ESP_LOGE("MAIN", "Failed to create sensor task");
        set_LED(255, 0, 0, 50);
        return;
    }
    
    // Small delay to let sensor task initialize
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Core 1: Classification task
    BaseType_t classification_result = xTaskCreatePinnedToCore(
        classification_task, "classification", 16384, NULL, 10, &classification_task_handle, 1);
    if (classification_result != pdPASS) {
        ESP_LOGE("MAIN", "Failed to create classification task");
        set_LED(255, 0, 0, 50);
        return;
    }

    ESP_LOGI("MAIN", "All tasks created successfully!");
    ESP_LOGI("MAIN", "Pipeline: IMU(10x) -> Camera -> Classification -> Repeat");
    ESP_LOGI("MAIN", "Core assignment: Sensor(Core 0), Classification(Core 1)");
    
    // // Monitor system health
    // for (;;) {
    //     vTaskDelay(pdMS_TO_TICKS(30000)); // Every 30 seconds
    //     ESP_LOGI("MAIN", "System health - Free heap: %lu bytes, Min free: %lu bytes",
    //              esp_get_free_heap_size(), esp_get_minimum_free_heap_size());
    // }
}   
