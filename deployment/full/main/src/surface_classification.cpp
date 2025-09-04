#include "surface_classification.hpp"
#include "BLEModule.h"

extern const uint8_t espdl_surface_model[] asm("_binary_surface_espdl_start");
dl::Model *surface_model = nullptr;
dl::image::ImagePreprocessor *m_surface_preprocessor = nullptr;

bool initialize_surface_model() {
    surface_model = new dl::Model((const char *)espdl_surface_model);
    if (!surface_model) {
        ESP_LOGE("SURFACE", "Failed to create model");
        return false;
    }

    m_surface_preprocessor = new dl::image::ImagePreprocessor(surface_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    if (!m_surface_preprocessor) {
        ESP_LOGE("SURFACE", "Failed to create image preprocessor");
        delete surface_model;
        surface_model = nullptr;
        return false;
    }

    return true;
}

bool convert_surface_image(const dl::image::img_t* input_img, dl::image::img_t &output_img) {
    // original height and width
    int orig_height = input_img->height;
    int orig_width = input_img->width;

    // crop to square
    int x_min = orig_width/6;
    int x_max = x_min + 96;
    int y_min = orig_height/4;
    int y_max = y_min + 96;
    std::vector<int> crop_area = {x_min, y_min, x_max, y_max};

    output_img.height = y_max-y_min;
    output_img.width = x_max-x_min;
    output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    output_img.data = malloc(output_img.height * output_img.width * 3); // RGB888: 3 bytes per pixel

    if (!output_img.data) {
        ESP_LOGE("SURFACE", "Memory allocation failed");
        free(output_img.data);
        return false;
    }

    // Convert using ESP-DL
    dl::image::convert_img(*input_img, output_img, 0, nullptr, crop_area);

    return true;
}

const std::vector<dl::cls::result_t> run_surface_inference(dl::image::img_t &input_img) {
    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();
    
    m_surface_preprocessor->preprocess(input_img);

    // TODO: try serializing model run with mutex
    surface_model->run();
    const int check = 5;
    SurfacePostProcessor m_postprocessor(surface_model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.\n", delta);

    for (auto &res : results) {
        ESP_LOGI("SURFACE", "category: %s, score: %f\n", res.cat_name, res.score);
    }

    return results;
}

bool process_surface_image(const dl::image::img_t* input_img) {
    dl::image::img_t converted_img;
    if (!convert_surface_image(input_img, converted_img)) {
        ESP_LOGE("SURFACE", "Could not convert image");
        return false;
    }

    const auto results = run_surface_inference(converted_img);

    float scores[5] = {
        results[0].score,
        results[1].score,
        results[2].score,
        results[3].score,
        results[4].score
    };

    notify_surface_classification(scores);
    
    // Free the original converted_img.data since it's been copied to ring buffer
    free(converted_img.data);
    converted_img.data = nullptr;

    return true;
}