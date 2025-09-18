#include "takeover_classification.hpp"
#include "BLEModule.h"

extern const uint8_t espdl_takeover_model[] asm("_binary_takeover_espdl_start");
dl::Model *takeover_model = nullptr;
dl::image::ImagePreprocessor *m_takeover_preprocessor = nullptr;

bool initialize_takeover_model() {    
    takeover_model = new dl::Model((const char *)espdl_takeover_model);
    if (!takeover_model) {
        ESP_LOGE("TAKEOVER", "Failed to create model");
        return false;
    }

    m_takeover_preprocessor = new dl::image::ImagePreprocessor(takeover_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    if (!m_takeover_preprocessor) {
        ESP_LOGE("TAKEOVER", "Failed to create image preprocessor");
        delete takeover_model;
        takeover_model = nullptr;
        return false;
    }

    return true;
}

bool convert_takeover_image(const dl::image::img_t* input_img, dl::image::img_t &output_img) {
    // original height and width
    int orig_height = input_img->height;
    int orig_width = input_img->width;

    // crop to square
    int x_min = orig_width-orig_height;
    int x_max = orig_width;
    int y_min = 0;
    int y_max = orig_height;
    std::vector<int> crop_area = {x_min, y_min, x_max, y_max};

    dl::image::img_t cropped_img;
    cropped_img.height = y_max-y_min;
    cropped_img.width = x_max-x_min;
    cropped_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    cropped_img.data = malloc(cropped_img.height * cropped_img.width * 3); // RGB888: 3 bytes per pixel

    if (!cropped_img.data) {
        ESP_LOGE("TAKEOVER", "Memory allocation failed");
        free(cropped_img.data);
        return false;
    }

    // Convert using ESP-DL
    dl::image::convert_img(*input_img, cropped_img, 0, nullptr, crop_area);

    // rescale to 24x24
    int target_w = 24;
    int target_h = 24;

    output_img.height = target_h;
    output_img.width = target_w;
    output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    output_img.data = malloc(target_h * target_w * 3); // RGB888: 3 bytes per pixel

    // Convert using ESP-DL
    dl::image::resize(cropped_img, output_img, dl::image::DL_IMAGE_INTERPOLATE_BILINEAR);

    free(cropped_img.data);

    return true;
}

std::vector<dl::cls::result_t> run_takeover_inference(const dl::image::img_t &input_img) {    
    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();
    
    m_takeover_preprocessor->preprocess(input_img);

    takeover_model->run(dl::RUNTIME_MODE_MULTI_CORE);
    const int check = 5;
    TakeoverPostProcessor m_postprocessor(takeover_model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    ESP_LOGI("TAKEOVER", "inference in %8.0f us.\n", delta);

    for (auto &res : results) {
        ESP_LOGI("TAKEOVER", "category: %s, score: %f\n", res.cat_name, res.score);
    }

    return results;
}

bool process_takeover_image(const dl::image::img_t* input_img) {
    const std::vector<dl::cls::result_t> results = run_takeover_inference(*input_img);

    float scores[1] = {0};
    for (const auto& res : results) {
        if (strcmp(res.cat_name, "takeover") == 0) {
            scores[0] = res.score;
        }
    }

    notify_takeover_classification(scores);

    return true;
}
