#include "surface_classification.hpp"

extern const uint8_t espdl_model[] asm("_binary_surface_espdl_start");

bool convert_surface_image(dl::image::img_t &input_img, dl::image::img_t &output_img) {
    // original height and width
    int orig_height = input_img.height;
    int orig_width = input_img.width;

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
        ESP_LOGE("MEM", "Memory allocation failed");
        free(output_img.data);
        return false;
    }

    return true;
}

const dl::cls::result_t run_surface_inference(dl::image::img_t &input_img) {
    char dir[64];
    snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    
    dl::Model *model = nullptr;
    dl::image::ImagePreprocessor *m_image_preprocessor = nullptr;
    
    model = new dl::Model((const char *)espdl_model, dir);
    if (!model) {
        ESP_LOGE("MODEL", "Failed to create model");
        return {};
    }
    
    // Add a small delay to ensure model is properly initialized
    vTaskDelay(pdMS_TO_TICKS(100));
    
    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();
    m_image_preprocessor = new dl::image::ImagePreprocessor(model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    if (!m_image_preprocessor) {
        ESP_LOGE("PREPROCESSOR", "Failed to create image preprocessor");
        delete model;
        return {};
    }
    
    m_image_preprocessor->preprocess(input_img);

    model->run();
    const int check = 5;
    SurfacePostProcessor m_postprocessor(model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.\n", delta);

    dl::cls::result_t best_result = {};
    bool found_result = false;

    for (auto &res : results) {
        ESP_LOGI("CLS", "category: %s, score: %f\n", res.cat_name, res.score);
        if (!found_result || res.score > best_result.score)
        {
            best_result = res;  // Copy the result
            found_result = true;
        }
    }
    
    // Free resources
    if (m_image_preprocessor) {
        delete m_image_preprocessor;
        m_image_preprocessor = nullptr;
    }
    if (model) {
        delete model;
        model = nullptr;
    }

    return best_result;
}

bool process_surface_image(dl::image::img_t &input_img, float &score, const char** category) {
    dl::image::img_t converted_img;
    if (!convert_surface_image(input_img, converted_img)) {
        ESP_LOGE("img", "Could not convert image");
        return false;
    }

    const auto best = run_surface_inference(converted_img);
    if (best.cat_name) {
        ESP_LOGI("INF", "Best: %s (score: %f)", best.cat_name, best.score);
    }
    
    score = best.score;
    *category = best.cat_name;
    
    // Free the original converted_img.data since it's been copied to ring buffer
    free(converted_img.data);
    converted_img.data = nullptr;

    return true;
}