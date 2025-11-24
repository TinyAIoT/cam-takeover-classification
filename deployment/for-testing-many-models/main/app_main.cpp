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
#include "include/ClassificationPostProcessor.hpp"
#include "include/classification_category_name.hpp"
#include "esp_camera.h"
// SD , SPI
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "esp_camera.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cstdio>
#include <esp_timer.h>
#include <stdarg.h>

// Support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif


// Linker-provided binary symbols (keep asm names matching linked symbols)
extern const uint8_t model_3x4x4[] asm("_binary_3x4x4_espdl_start");
extern const uint8_t model_1x4x4[] asm("_binary_1x4x4_espdl_start");
extern const uint8_t model_3x1x16[] asm("_binary_3x1x16_espdl_start");
extern const uint8_t model_1x1x16[] asm("_binary_1x1x16_espdl_start");
static const char *model_3x4x4_path = (const char *)model_3x4x4;
static const char *model_1x4x4_path = (const char *)model_1x4x4;
static const char *model_3x1x16_path = (const char *)model_3x1x16;
static const char *model_1x1x16_path = (const char *)model_1x1x16;
// List of available models with friendly IDs
struct ModelEntry {
    const char *id;
    const char *path;
};

static const ModelEntry available_models[] = {
    { "3x4x4",      model_3x4x4_path },
    { "1x4x4",      model_1x4x4_path },
    { "3x1x16",     model_3x1x16_path },
    { "1x1x16",     model_1x1x16_path },
};

static const size_t available_model_count = sizeof(available_models) / sizeof(available_models[0]);
dl::Model *takeover_model = nullptr;
dl::image::ImagePreprocessor *m_takeover_preprocessor = nullptr;

bool initialize_model(const char *model_path) {    
    takeover_model = new dl::Model(model_path, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    if (!takeover_model) {
        ESP_LOGE("TAKEOVER", "Failed to create model");
        return false;
    }

    if (takeover_model->get_input("")->shape[3] == 3) {
        m_takeover_preprocessor = new dl::image::ImagePreprocessor(takeover_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    } else if (takeover_model->get_input("")->shape[3] == 1) {
        m_takeover_preprocessor = new dl::image::ImagePreprocessor(takeover_model, {123.675}, {58.395});
    } else {
        ESP_LOGE("TAKEOVER", "Unsupported number of channels: %d", takeover_model->get_input("")->shape[3]);
        delete takeover_model;
        takeover_model = nullptr;
        return false;
    }
    if (!m_takeover_preprocessor) {
        ESP_LOGE("TAKEOVER", "Failed to create image preprocessor");
        delete takeover_model;
        takeover_model = nullptr;
        return false;
    }
    // NOTE: profiling (profile_memory/profile_module) is intentionally not
    // called here because the caller may want to capture the logging output
    // produced by those calls and write it to a file. Caller should invoke
    // profiling after initialize_model() and while a log-capture hook is active.

    return true;
}

#include "sd_pins.h"
#define MOUNT_POINT "/sdcard"
sdmmc_card_t *g_card;
// Log capture plumbing: we temporarily replace the esp log vprintf with a
// function that also writes the formatted output to a FILE* (when set), and
// forwards to the original vprintf implementation so logs still appear on
// console/UART.
static FILE *g_log_capture_file = nullptr;
static int (*g_orig_vprintf)(const char *, va_list) = nullptr;

static int log_capture_vprintf(const char *fmt, va_list ap)
{
    int ret = 0;
    if (g_log_capture_file) {
        va_list ap_copy;
        va_copy(ap_copy, ap);
        vfprintf(g_log_capture_file, fmt, ap_copy);
        va_end(ap_copy);
        fflush(g_log_capture_file);
    }
    if (g_orig_vprintf) {
        va_list ap_copy2;
        va_copy(ap_copy2, ap);
        ret = g_orig_vprintf(fmt, ap_copy2);
        va_end(ap_copy2);
    } else {
        // Fallback to libc vprintf if original is not available
        va_list ap_copy3;
        va_copy(ap_copy3, ap);
        ret = vprintf(fmt, ap_copy3);
        va_end(ap_copy3);
    }
    return ret;
}

static void start_log_capture(FILE *f)
{
    if (!f) return;
    if (!g_orig_vprintf) {
        g_orig_vprintf = esp_log_set_vprintf(log_capture_vprintf);
    } else {
        esp_log_set_vprintf(log_capture_vprintf);
    }
    g_log_capture_file = f;
}

static void stop_log_capture(void)
{
    // restore original vprintf if we stored it
    if (g_orig_vprintf) {
        esp_log_set_vprintf(g_orig_vprintf);
    }
    g_log_capture_file = nullptr;
}
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
    ESP_LOGI("MEM", "Free heap at A: %lu bytes", esp_get_free_heap_size());
    // gpio_set_level(SD_ENABLE, 1);
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &g_card);

    ESP_LOGI("MEM", "Free heap at B: %lu bytes", esp_get_free_heap_size());
    if (ret != ESP_OK)
    {
    ESP_LOGI("MEM", "Free heap at C: %lu bytes", esp_get_free_heap_size());
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


extern "C" void app_main(void) {
    // Mount SD card first (we need it for saving results)
    if (!mount_sdcard_spi()) {
        ESP_LOGE("SD", "Failed to mount SD card. Aborting runs that write results to SD.");
        return;
    }

    // Create output directory on SD card
    const char *out_dir = MOUNT_POINT "/model_runs";
    createDir(out_dir);

    // Prepare permutations of model indices. We'll run up to 24 unique orders.
    size_t N = available_model_count;
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<std::vector<int>> orders;

    // Use next_permutation to generate deterministic permutations and take first 24 (or all if fewer)
    std::sort(idx.begin(), idx.end());
    size_t max_runs = 24;
    do {
        orders.push_back(idx);
        if (orders.size() >= max_runs) break;
    } while (std::next_permutation(idx.begin(), idx.end()));

    ESP_LOGI("RUNS", "Prepared %d permutation runs", (int)orders.size());

    // Helper to cleanup currently loaded model
    auto cleanup_model = []() {
        if (m_takeover_preprocessor) {
            delete m_takeover_preprocessor;
            m_takeover_preprocessor = nullptr;
        }
        if (takeover_model) {
            delete takeover_model;
            takeover_model = nullptr;
        }
    };

    // For each run, load models in the specified order and record metrics
    for (size_t run = 0; run < orders.size(); ++run) {
        // Build filename for this run
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/runs_%02d.txt", out_dir, (int)run);
        FILE *f = fopen(fname, "w");
        if (!f) {
            ESP_LOGE("FILE", "Failed to open file for writing: %s", fname);
            // continue but skip writing
        } else {
            fprintf(f, "Model run %02d\n", (int)run);
            fprintf(f, "Order:");
            for (int id : orders[run]) {
                fprintf(f, " %s", available_models[id].id);
            }
            fprintf(f, "\n\n");
        }

        for (size_t pos = 0; pos < orders[run].size(); ++pos) {
            int mid = orders[run][pos];
            const char *mid_id = available_models[mid].id;
            const char *m_path = available_models[mid].path;

            ESP_LOGI("RUN", "Run %d, pos %d -> Model %s", (int)run, (int)pos, mid_id);

            // ensure previous model unloaded
            cleanup_model();

            size_t heap_before = esp_get_free_heap_size();
            int64_t t0 = esp_timer_get_time();
            bool ok = initialize_model(m_path);
            int64_t t1 = esp_timer_get_time();
            size_t heap_after = esp_get_free_heap_size();

            int64_t init_ms = (t1 - t0) / 1000;

            ESP_LOGI("METRIC", "Model %s init %s ms, heap_before=%u, heap_after=%u", mid_id, (init_ms>=0?"":""), (unsigned)heap_before, (unsigned)heap_after);

            if (f) {
                fprintf(f, "Model: %s\n", mid_id);
                fprintf(f, "  init_success: %s\n", ok ? "true" : "false");
                fprintf(f, "  init_time_ms: %lld\n", (long long)init_ms);
                fprintf(f, "  heap_before: %u\n", (unsigned)heap_before);
                fprintf(f, "  heap_after: %u\n", (unsigned)heap_after);
                fprintf(f, "\n");
            }

            // Capture the textual output produced by profile_memory/profile_module
            if (ok) {
                if (f) fprintf(f, "--- PROFILE OUTPUT START ---\n");
                start_log_capture(f);
                // These calls produce textual logging; our vprintf hook writes it to the file
                takeover_model->profile_memory();
                takeover_model->profile_module();
                stop_log_capture();
                if (f) fprintf(f, "--- PROFILE OUTPUT END ---\n\n");
                if (f) fflush(f);
            }

            // unload to simulate fresh start for next model
            cleanup_model();
        }

        if (f) {
            fclose(f);
            ESP_LOGI("FILE", "Wrote results to %s", fname);
        }
    }

    // Final cleanup
    // (ensure no model left allocated)
    if (m_takeover_preprocessor || takeover_model) {
        if (m_takeover_preprocessor) delete m_takeover_preprocessor;
        if (takeover_model) delete takeover_model;
        m_takeover_preprocessor = nullptr;
        takeover_model = nullptr;
    }
}
