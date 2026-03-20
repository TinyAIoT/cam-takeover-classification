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
#include "include/dci_calculator.h"

#include "include/imu.h"

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
#define IMU_SAMPLING_INTERVAL_MS 10 // 100 Hz sampling rate

// Image buffers for double-buffering
static dl::image::img_t surface_buf[2]; // allocate these properly (global/static/heap)
static volatile int surface_read_idx = 0;
static int surface_write_idx = 1;
static portMUX_TYPE surface_buf_spinlock = portMUX_INITIALIZER_UNLOCKED;

// IMU data collection
static imu_data_t imu_buffer[MIN_IMU_READINGS];
static volatile int imu_count = 0;
static SemaphoreHandle_t imu_ready_sem;

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
    .fb_count = 1, // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = 0 // optional
};

// Helper function for surface image conversion from camera frame buffer
bool convert_surface_image_from_fb(camera_fb_t *fb, dl::image::img_t &output_img)
{
    // Create temporary image from frame buffer
    dl::image::img_t temp_img = {
        .data = fb->buf,
        .width = (uint16_t)fb->width,
        .height = (uint16_t)fb->height,
        .pix_type = (camera_config.pixel_format == PIXFORMAT_GRAYSCALE) ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB565};

    // Convert using your existing function
    return convert_surface_image(&temp_img, output_img,
                                 camera_config.pixel_format == PIXFORMAT_GRAYSCALE ? dl::image::DL_IMAGE_PIX_TYPE_GRAY : dl::image::DL_IMAGE_PIX_TYPE_RGB888);
}

static esp_err_t init_camera(void)
{
    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK)
    {
        ESP_LOGE("CAM", "Camera Init Failed");
    }
    return err;
}

static bool capture_and_process_image(dl::image::img_t &output_img)
{
    ESP_LOGD("CAM", "Capturing picture...");
    camera_fb_t *pic = esp_camera_fb_get();
    if (!pic)
    {
        ESP_LOGE("CAM", "Failed to capture image");
        return false;
    }

    // Validate frame buffer
    if (!pic->buf || pic->len == 0)
    {
        ESP_LOGE("CAM", "Invalid frame buffer: buf=%p, len=%zu", pic->buf, pic->len);
        esp_camera_fb_return(pic);
        return false;
    }

    // Use pic->buf to access the image
    ESP_LOGD("CAM", "Picture taken! Height: %d, Width: %d, Len: %zu", pic->height, pic->width, pic->len);

    // Free previous image data (if any)
    if (output_img.data)
    {
        free(output_img.data);
        output_img.data = nullptr;
    }

    // Convert and crop for surface classification
    if (!convert_surface_image_from_fb(pic, output_img))
    {
        ESP_LOGE("CAM", "Surface conversion failed");
        esp_camera_fb_return(pic);
        return false;
    }

    esp_camera_fb_return(pic);
    return true;
}

// Initialize buffers properly
static void init_buffers()
{
    for (int i = 0; i < 2; i++)
    {
        // memset(&g_buf[i], 0, sizeof(dl::image::img_t));
        memset(&surface_buf[i], 0, sizeof(dl::image::img_t));
    }
}

// Add these timing variables at the top of the file
static struct
{
    uint64_t camera_capture_us;
    uint64_t image_processing_us;
    uint64_t buffer_swap_us;
    uint64_t notification_us;
    uint64_t total_imu_inactive_us;
} timing_stats = {0};

static void sensor_task(void *pvParameters)
{
    ESP_LOGI("SENSOR", "Sensor task started on core %d", xPortGetCoreID());

    // Initialize IMU
    if (imu_init() != ESP_OK)
    {
        ESP_LOGE("SENSOR", "Failed to initialize IMU");
        set_LED(255, 0, 0, 50);
        vTaskDelete(NULL);
        return;
    }

    imu_data_t imu_data;
    int cycle_count = 0;

    for (;;)
    {
        // Phase 1: Collect IMU readings
        ESP_LOGD("SENSOR", "=== Cycle %d: Collecting IMU readings ===", ++cycle_count);

        uint64_t imu_collection_start = esp_timer_get_time();

        imu_count = 0;
        TickType_t last_wake_time = xTaskGetTickCount();
        TickType_t sampling_period = pdMS_TO_TICKS(IMU_SAMPLING_INTERVAL_MS);

        for (int i = 0; i < MIN_IMU_READINGS; i++)
        {
            // Set sampling interval to 10ms (100Hz)
            vTaskDelayUntil(&last_wake_time, sampling_period);

            if (imu_read(&imu_data) == ESP_OK && imu_data.valid)
            {
                memcpy(&imu_buffer[i], &imu_data, sizeof(imu_data_t));
                imu_count++;
                imu_fps.update();

                ESP_LOGD("SENSOR", "IMU %d/%d: Acc[%d,%d,%d] Gyr[%d,%d,%d]",
                         i + 1, MIN_IMU_READINGS,
                         imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
                         imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z);
            }
            else
            {
                ESP_LOGW("SENSOR", "Failed to read IMU data %d/%d", i + 1, MIN_IMU_READINGS);
                i--; // Retry this reading
            }
        }

        uint64_t imu_collection_end = esp_timer_get_time();
        uint64_t imu_collection_time = imu_collection_end - imu_collection_start;

        ESP_LOGD("SENSOR", "Collected %d IMU readings in %llu ms",
                 imu_count, imu_collection_time / 1000);

        // Phase 2: Signal camera & classification task and continue IMU collection
        if (classification_task_handle && classification_ready)
        {
            ESP_LOGD("SENSOR", "=== Phase 2: Notifying classification task ===");
            BaseType_t result = xTaskNotifyGive(classification_task_handle);
            if (result != pdPASS)
            {
                ESP_LOGE("SENSOR", "Failed to notify classification task!");
            }
        }
        else
        {
            ESP_LOGW("SENSOR", "Classification not ready: handle=%p, ready=%d",
                     classification_task_handle, classification_ready);
        }

        // Phase 1.5: Calculate DCI from collected IMU data (simplified)
        if (imu_count >= MIN_IMU_READINGS)
        {
            dci_result_t dci_result;
            float gravity_ref = 1.0f; // Assume 1g for simplicity, or calculate from data if needed
            uint64_t dci_calculate_start = esp_timer_get_time();
            esp_err_t dci_ret = dci_calculate(imu_buffer, imu_count, gravity_ref, &dci_result);
            uint64_t dci_calculate_end = esp_timer_get_time();
            // Log timing information in microseconds for more precision
            ESP_LOGD("SENSOR", "DCI calculation took %llu us", dci_calculate_end - dci_calculate_start);
            if (dci_ret == ESP_OK && dci_result.valid)
            {
                ESP_LOGD("SENSOR", "DCI Analysis (Z-axis downward):");
                ESP_LOGI("SENSOR", "  - DCI Value: %.3f", dci_result.dci_value);
                ESP_LOGD("SENSOR", "  - Samples: %d total, %d above threshold (1g)",
                         dci_result.total_samples, dci_result.above_threshold);
                ESP_LOGD("SENSOR", "  - Surface roughness: %s",
                         dci_result.dci_value > 2.0f ? "Smooth" : dci_result.dci_value > 1.0f ? "Moderate"
                                                                                              : "Rough");
            }
            else
            {
                ESP_LOGW("SENSOR", "DCI calculation failed: not enough dynamic acceleration");
            }
        }

        // Small delay before next cycle
        // vTaskDelay(pdMS_TO_TICKS(10));
    }
}

static void classification_task(void *pvParameters)
{
    ESP_LOGI("CLASSIFY", "start (core=%d, tick=%u)", xPortGetCoreID(), xTaskGetTickCount());

    if (!initialize_surface_model())
    {
        set_LED(255, 0, 150, 20);
        ESP_LOGE("SURFACE", "Failed to initialize surface model");
        vTaskDelete(NULL);
        return;
    }

    // Signal that classification is ready
    classification_ready = true;
    ESP_LOGD("CLASSIFY", "Surface model initialized, ready for classification");

    int iteration_count = 0;

    for (;;)
    {
        // Wait for notification from camera task that a new frame is available
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        ESP_LOGD("CLASSIFY", "=== Camera capture phase ===");

        // Sub-phase 2a: Camera capture
        uint64_t camera_start = esp_timer_get_time();

        bool camera_success = capture_and_process_image(surface_buf[surface_write_idx]);

        uint64_t camera_end = esp_timer_get_time();

        timing_stats.camera_capture_us = camera_end - camera_start;

        if (!camera_success)
        {
            ESP_LOGE("CLASSIFY", "Camera capture failed after %llu ms, retrying...",
                     timing_stats.camera_capture_us / 1000);
            continue;
        }

        camera_fps.update();

        ESP_LOGD("CLASSIFY", "Camera capture completed in %llu ms",
                 timing_stats.camera_capture_us / 1000);

        // Phase 3: Publish image for classification (atomic swap)
        portENTER_CRITICAL(&surface_buf_spinlock);
        surface_read_idx = surface_write_idx;
        surface_write_idx ^= 1; // Flip between 0 and 1
        portEXIT_CRITICAL(&surface_buf_spinlock);

        iteration_count++;
        classification_fps.update();

        ESP_LOGD("CLASSIFY", "=== Classification iteration %d ===", iteration_count);

        // Get current buffer index atomically
        int idx;
        portENTER_CRITICAL(&surface_buf_spinlock);
        idx = surface_read_idx;
        portEXIT_CRITICAL(&surface_buf_spinlock);

        // Validate buffer before processing
        if (!surface_buf[idx].data)
        {
            ESP_LOGW("CLASSIFY", "No valid data in buffer %d", idx);
            continue;
        }

        // Process surface classification
        TickType_t start_time = xTaskGetTickCount();
        bool success = process_surface_image(&surface_buf[idx]);
        TickType_t end_time = xTaskGetTickCount();

        uint32_t elapsed_ms = (end_time - start_time) * portTICK_PERIOD_MS;

        if (!success)
        {
            ESP_LOGE("CLASSIFY", "Surface processing failed (took %lu ms)", elapsed_ms);
            set_LED(255, 0, 0, 30); // Red - error
        }
        else
        {
            ESP_LOGD("CLASSIFY", "Surface processing succeeded (took %lu ms)", elapsed_ms);
            set_LED(0, 255, 0, 10); // Green - success
        }

        // Log performance periodically
        if (iteration_count % 5 == 0)
        {
            ESP_LOGD("CLASSIFY", "Performance - Cam: %.1f fps, IMU: %.1f fps, Classify: %.1f fps",
                     camera_fps.get_current(), imu_fps.get_current(), classification_fps.get_current());
        }
    }
}

extern "C" void app_main(void)
{
    // Initialize LED and show startup
    init_LED();
    set_LED(255, 255, 0, 20);

    init_buffers();

    // Create semaphores
    imu_ready_sem = xSemaphoreCreateBinary();
    if (imu_ready_sem == NULL)
    {
        ESP_LOGE("MAIN", "Failed to create IMU ready semaphore");
        set_LED(255, 0, 0, 50); // Red - fatal error
        return;
    }

    // Only show warnings and errors at runtime (suppress INFO/DEBUG)
    // Set logging levels
    // esp_log_level_set("*", ESP_LOG_INFO);
    // esp_log_level_set("SENSOR", ESP_LOG_DEBUG);
    // esp_log_level_set("CLASSIFY", ESP_LOG_DEBUG);

    // Initialize NVS (if not already done)
    esp_err_t nvs_ret = nvs_flash_init();
    if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES || nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_ret);

    esp_err_t ret = nimble_port_init();
    if (ret != ESP_OK)
    {
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

    if (ESP_OK != init_camera())
    {
        ESP_LOGE("CAM", "Camera init failed");
        set_LED(255, 50, 0, 20);
        return;
    }

    set_LED(0, 255, 0, 10);

    // Create tasks
    ESP_LOGI("MAIN", "Creating tasks...");

    // Core 0: Sensor task (IMU + Camera)
    BaseType_t sensor_result = xTaskCreatePinnedToCore(
        sensor_task, "sensor", 12288, NULL, 15, &sensor_task_handle, 1);
    if (sensor_result != pdPASS)
    {
        ESP_LOGE("MAIN", "Failed to create sensor task");
        set_LED(255, 0, 0, 50);
        return;
    }

    // Small delay to let sensor task initialize
    vTaskDelay(pdMS_TO_TICKS(1000));

    // Core 1: Classification task
    BaseType_t classification_result = xTaskCreatePinnedToCore(
        classification_task, "classification", 16384, NULL, 10, &classification_task_handle, 0);
    if (classification_result != pdPASS)
    {
        ESP_LOGE("MAIN", "Failed to create classification task");
        set_LED(255, 0, 0, 50);
        return;
    }

    ESP_LOGI("MAIN", "All tasks created successfully!");
    ESP_LOGI("MAIN", "Pipeline: \n Core 1: IMU(20x) -> Notify Core 0 -> Repeat \n Core 0: Wait for Notification -> Camera -> Classification -> Repeat");
    ESP_LOGI("MAIN", "Core assignment: Sensor(Core 1), Camera + Classification(Core 0)");

}
