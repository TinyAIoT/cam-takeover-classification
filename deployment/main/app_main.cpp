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
#include "include/BirdPostProcessor.hpp"
#include "include/bird_category_name.hpp"
#include "esp_camera.h"
// SD , SPI
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "esp_camera.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

const char *TAG2 = "bird_cls";


extern const uint8_t birb_jpg_start[] asm("_binary_bluetit_jpg_start");
extern const uint8_t birb_jpg_end[] asm("_binary_bluetit_jpg_end");
extern const uint8_t espdl_model[] asm("_binary_torch_mbnv2_layerwise_equalization_espdl_start");

// Set to true to take a camera picture, else make sure to add an img
#define TAKE_PICTURE true
#define SAVE_TO_SDCARD true

#if SAVE_TO_SDCARD
#include "sd_pins.h"
#endif // SAVE_TO_SDCARD
#define MOUNT_POINT "/sdcard"

#if TAKE_PICTURE && ESP_CAMERA_SUPPORTED
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

    // dl::image::jpeg_img_t jpeg_img = {
    //     .data = (uint8_t *)pic->buf,
    //     .width = (int)pic->width,
    //     .height = (int)pic->height,
    //     .data_size = (uint32_t)(pic->len),
    // };

    // Prepare ESP-DL image structs
    dl::image::img_t rgb565_img;
    rgb565_img.data = pic->buf;
    rgb565_img.height = pic->height;
    rgb565_img.width = pic->width;
    rgb565_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;

    output_img.height = pic->height;
    output_img.width = pic->width;
    output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    output_img.data = malloc(pic->height * pic->width * 3); // RGB888: 3 bytes per pixel

    if (!output_img.data) {
        ESP_LOGE("MEM", "Memory allocation failed");
        esp_camera_fb_return(pic);
        return false;
    }

    // Convert using ESP-DL
    int x_min = 0;
    int x_max = 160;
    int y_min = 0;
    int y_max = 160;
    std::vector<int> crop_area = {x_min, y_min, x_max, y_max};
    dl::image::convert_img(rgb565_img, output_img, 0, nullptr, crop_area);

    esp_camera_fb_return(pic);
    return true;
}
#endif // TAKE_PICTURE && ESP_CAMERA_SUPPORTED

static const dl::cls::result_t run_inference(dl::image::img_t &input_img) {
    char dir[64];
    // TODO: as we are testing multiple models we might want to include them in a smarter way. Is there something like command line arguments?
    snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    dl::Model *model = new dl::Model((const char *)espdl_model, dir);

    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();

    dl::image::ImagePreprocessor *m_image_preprocessor = new dl::image::ImagePreprocessor(model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    m_image_preprocessor->preprocess(input_img);

    model->run();
    const int check = 5;
    BirdPostProcessor m_postprocessor(model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.", delta);

    dl::cls::result_t best_result = {};
    bool found_result = false;

    for (auto &res : results) {
        ESP_LOGI(TAG2, "category: %s, score: %f\n", res.cat_name, res.score);
        if (!found_result || res.score > best_result.score)
        {
            best_result = res;  // Copy the result
            found_result = true;
        }
    }

    // Free resources
    if (model) {
        delete model;
        model = nullptr;
    }
    if (m_image_preprocessor) {
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
    }

    return best_result;
}

// **********************
sdmmc_card_t *g_card;
int count_files_in_directory(const char *path)
{
    int count = 0;
    DIR *dir = opendir(path);
    if (!dir)
    {
        ESP_LOGE("FILE_COUNT", "Failed to open directory: %s", path);
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        // Skip current and parent directory entries
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }

        // Build full file path
        char full_path[256];
        // snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        strlcpy(full_path, path, sizeof(full_path));
        strlcat(full_path, "/", sizeof(full_path));
        strlcat(full_path, entry->d_name, sizeof(full_path));
        // Get file info
        struct stat st;
        if (stat(full_path, &st) == 0)
        {
            if (S_ISREG(st.st_mode))
            {
                count++;
            }
        }
        else
        {
            ESP_LOGW("FILE_COUNT", "Could not stat file: %s", full_path);
        }
    }

    closedir(dir);
    return count;
}
void createDir(const char *path)
{
    static const char *TAG = "createDir";
    ESP_LOGI(TAG, "Creating Dir: %s", path);

    FRESULT res = f_mkdir(path);
    if (res == FR_OK)
    {
        ESP_LOGI(TAG, "Dir created");
    }
    else if (res == FR_EXIST)
    {
        ESP_LOGI(TAG, "Dir already exists");
    }
    else
    {
        ESP_LOGE(TAG, "mkdir failed with error: %d", res);
    }
}
void init_sd_enable_pin(void)
{
    // Configure the GPIO as output
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << SD_ENABLE),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE};
    gpio_config(&io_conf);

    // Set the output level
    gpio_set_level(SD_ENABLE, 0);
}
bool mount_sdcard_spi()
{
    init_sd_enable_pin();
    esp_err_t ret;

    // Options for mounting the filesystem.
    // If format_if_mount_failed is set to true, SD card will be partitioned and
    // formatted in case when mounting fails.
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
#ifdef CONFIG_EXAMPLE_FORMAT_IF_MOUNT_FAILED
        .format_if_mount_failed = true,
#else
        .format_if_mount_failed = false,
#endif // EXAMPLE_FORMAT_IF_MOUNT_FAILED
        .max_files = 5,
        .allocation_unit_size = 16 * 1024};
    // sdmmc_card_t *card;
    const char mount_point[] = MOUNT_POINT;
    ESP_LOGI("SD", "Initializing SD card");

    // Use settings defined above to initialize SD card and mount FAT filesystem.
    // Note: esp_vfs_fat_sdmmc/sdspi_mount is all-in-one convenience functions.
    // Please check its source code and implement error recovery when developing
    // production applications.
    ESP_LOGI("SD", "Using SPI peripheral");

    // By default, SD card frequency is initialized to SDMMC_FREQ_DEFAULT (20MHz)
    // For setting a specific frequency, use host.max_freq_khz (range 400kHz - 20MHz for SDSPI)
    // Example: for fixed frequency of 10MHz, use host.max_freq_khz = 10000;
    // host.'slot' should be set to an sdspi device initialized by `sdspi_host_init_device()`.
    // SDSPI_HOST_DEFAULT: https://github.com/espressif/esp-idf/blob/1bbf04cb4cf54d74c1fe21ed12dbf91eb7fb1019/components/esp_driver_sdspi/include/driver/sdspi_host.h#L44
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.max_freq_khz = 5000;
#define SPI_HOST_ID SPI3_HOST // #if SOC_SPI_PERIPH_NUM > 2 ???

    host.slot = SPI_HOST_ID; //

// For SoCs where the SD power can be supplied both via an internal or external (e.g. on-board LDO) power supply.
// When using specific IO pins (which can be used for ultra high-speed SDMMC) to connect to the SD card
// and the internal LDO power supply, we need to initialize the power supply first.
#if CONFIG_EXAMPLE_SD_PWR_CTRL_LDO_INTERNAL_IO
    sd_pwr_ctrl_ldo_config_t ldo_config = {
        .ldo_chan_id = CONFIG_EXAMPLE_SD_PWR_CTRL_LDO_IO_ID,
    };
    sd_pwr_ctrl_handle_t pwr_ctrl_handle = NULL;

    ret = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &pwr_ctrl_handle);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to create a new on-chip LDO power control driver");
        return false;
    }
    host.pwr_ctrl_handle = pwr_ctrl_handle;
#endif
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    ESP_LOGI("spi", "SDSPI_DEFAULT_DMA: %d", SDSPI_DEFAULT_DMA);
    ret = spi_bus_initialize(SPI_HOST_ID, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (ret != ESP_OK)
    {
        ESP_LOGE("SD", "Failed to initialize bus (ret != ESP_OK).");

        return false;
    }
    // card select output ?
    gpio_reset_pin(PIN_NUM_CS);
    gpio_set_direction(PIN_NUM_CS, GPIO_MODE_OUTPUT);
    gpio_set_level(PIN_NUM_CS, 1); // Inaktiv

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    // sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    sdspi_device_config_t slot_config = {
        .host_id = SPI_HOST_ID,
        .gpio_cs = PIN_NUM_CS,
        .gpio_cd = SD_SW,
        .gpio_wp = SDSPI_SLOT_NO_WP,
        .gpio_int = GPIO_NUM_NC,
        .gpio_wp_polarity = SDSPI_IO_ACTIVE_LOW,
        //.duty_cycle_pos = 0,
    };
    // spi_host_device_t host_id; ///< SPI host to use, SPIx_HOST (see spi_types.h)
    ESP_LOGI("SD", "Mounting filesystem");
    // gpio_set_level(SD_ENABLE, 1);
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &g_card);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE("SD", "Failed to mount filesystem (ret == ESP_FAIL). Look into esp_vfs_fat_sdspi_mount() "
                          "If you want the card to be formatted, set the CONFIG_EXAMPLE_FORMAT_IF_MOUNT_FAILED menuconfig option.");
        }
        else
        {
            ESP_LOGE("SD", "Failed to initialize the card (%s). Look into esp_vfs_fat_sdspi_mount() "
                          "Make sure SD card lines have pull-up resistors in place.",
                     esp_err_to_name(ret));
#ifdef CONFIG_EXAMPLE_DEBUG_PIN_CONNECTIONS
            check_sd_card_pins(&config, pin_count);
#endif
        }
        return false;
    }
    ESP_LOGI("SD", "Filesystem mounted");

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, g_card);
    return true;
};
jpeg_error_t encode_img_to_jpeg(dl::image::img_t *img, dl::image::jpeg_img_t *jpeg_img, jpeg_enc_config_t jpeg_enc_cfg)
{
    jpeg_enc_handle_t jpeg_enc = NULL;
    jpeg_error_t ret = jpeg_enc_open(&jpeg_enc_cfg, &jpeg_enc);
    if (ret != JPEG_ERR_OK)
    {
        return ret;
    }

    int outbuf_size = 100 * 1024; // 100 KB
    uint8_t *outbuf = (uint8_t *)calloc(1, outbuf_size);
    if (!outbuf)
    {
        jpeg_enc_close(jpeg_enc);
        return JPEG_ERR_NO_MEM;
    }

    int out_len = 0;
    ret = jpeg_enc_process(jpeg_enc, (const uint8_t *)img->data, img->width * img->height * 3, outbuf, outbuf_size, &out_len);
    if (ret == JPEG_ERR_OK)
    {
        jpeg_img->data = outbuf;
        jpeg_img->data_size = out_len;
    }
    else
    {
        free(outbuf);
    }

    jpeg_enc_close(jpeg_enc);
    return ret;
}
// ***********************

extern "C" void app_main(void) {
    dl::image::img_t img;

    bool mounted = false;
    ESP_LOGI("SD", "Mounting SD card...");
    mounted = mount_sdcard_spi();
    vTaskDelay(pdMS_TO_TICKS(1000));

#if TAKE_PICTURE && ESP_CAMERA_SUPPORTED
    if (ESP_OK != init_camera())
    {
        ESP_LOGE("SD", "Camera initialization failed. Look into init_camera()");
        return;
    }
    ESP_LOGI("SD", "Free heap before make dirs: %lu bytes", esp_get_free_heap_size());
    // make directories for each category
    for (int i = 0; i < sizeof(bird_cat_names) / sizeof(bird_cat_names[0]); i++)
    {
        char dirpath[128];
        snprintf(dirpath, sizeof(dirpath), "%s/%s", MOUNT_POINT, bird_cat_names[i]);
        createDir(bird_cat_names[i]);
        vTaskDelay(pdMS_TO_TICKS(100)); // Delay to ensure directory creation
    }

    while (mounted)
    {
        if (!capture_image(img)) {
            ESP_LOGE("img", "Could not take picture");
            return;
        }
        
        const auto best = run_inference(img);
        if (best.cat_name) {
            ESP_LOGI("INF", "Best: %s (score: %f)", best.cat_name, best.score);
        }

        dl::image::jpeg_img_t encoded_jpeg_img; // Für das JPEG-Ergebnis
        // JPEG enkodieren
        jpeg_error_t encode_ret;
        jpeg_enc_config_t enc_config = {
            .width = img.width,
            .height = img.height,
            .src_type = JPEG_PIXEL_FORMAT_RGB888,
            .subsampling = JPEG_SUBSAMPLE_444, // JPEG_SUBSAMPLE_420
            .quality = 80,
            .rotate = JPEG_ROTATE_0D,
            .task_enable = true,
            .hfm_task_priority = 13,
            .hfm_task_core = 1,

        };

        encode_ret = encode_img_to_jpeg(&img, &encoded_jpeg_img, enc_config);

        if (encode_ret != JPEG_ERR_OK)
        {
            ESP_LOGE("JPEG", "JPEG Encoding failed!");
        }
        else
        {

            if (img.data)
            {

                heap_caps_free(img.data);
                img.data = nullptr;
            }
            // JPEG-Daten in Datei speichern

            esp_err_t write_jpeg_err;

            char filename[64];
            const char *label = best.cat_name;
// float score = best->score;
#if SAVE_TO_SDCARD

            char dir_path[128];
            char filepath[128];
            snprintf(dir_path, sizeof(dir_path), "/sdcard/%s", label);

            int file_count = count_files_in_directory(dir_path);
            snprintf(filename, sizeof(filename), "%s_%.4f_%d.jpg", label, best.score, file_count + 1);
            ESP_LOGI("SD", "Saving image to SD card as %s", filename);
            snprintf(filepath, sizeof(filepath), "/sdcard/%s/%s", label, filename);
            ESP_LOGI("SD", "Writing JPEG. Filepath: %s", filepath);
            write_jpeg_err = dl::image::write_jpeg(encoded_jpeg_img, filepath);
            if (write_jpeg_err == ESP_OK)
            {
                ESP_LOGI("SD", "JPEG written to SD card successfully!");
            }
            if (write_jpeg_err != ESP_OK)
            {

                ESP_LOGE("SD", "Failed to write JPEG to SD card!");
                // esp_vfs_fat_sdcard_unmount(MOUNT_POINT, g_card);
                // ESP_LOGI(TAG, "Card unmounted");
                // spi_bus_free(SPI_HOST_ID);
                //  restart if jpeg writing failed
                ESP_LOGI("SD", "Restarting loop due to write error.");
                continue;
            }
            // Speicher des JPEG-Ergebnisses freigeben
            heap_caps_free(encoded_jpeg_img.data);
        }
#endif // SAVE_TO_SDCARD

        heap_caps_free(img.data);
        ESP_LOGI("MEM", "Free heap at end of loop: %lu bytes", esp_get_free_heap_size());
    }

    
#else
    // If an example image is used
    dl::image::jpeg_img_t jpeg_img = {
        .data = (uint8_t *)birb_jpg_start,
        .width = 160,
        .height = 160,
        .data_size = (uint32_t)(birb_jpg_end - birb_jpg_start),
    };
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    sw_decode_jpeg(jpeg_img, img, true);

    const auto best = run_inference(img);
    if (best.cat_name) {
        ESP_LOGI(TAG2, "Best: %s (score: %f)", best.cat_name, best.score);
    }

    heap_caps_free(img.data);
    ESP_LOGI("MEM", "Free heap at end of loop: %lu bytes", esp_get_free_heap_size());
#endif
}
