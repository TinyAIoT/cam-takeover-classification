#include "surface_classification.hpp"
#include "BLEModule.h"

extern const uint8_t espdl_surface_model[] asm("_binary_surface_espdl_start");
dl::Model *surface_model = nullptr;
dl::image::ImagePreprocessor *m_surface_preprocessor = nullptr;
dl::image::ImageTransformer surfaceTransformer;

bool initialize_surface_model() {
    surface_model = new dl::Model((const char *)espdl_surface_model);
    if (!surface_model) {
        ESP_LOGE("SURFACE", "Failed to create model");
        return false;
    }
    
    // TODO: the preprocesser could potentially be recreated by us, to save a bit of computation. Because we are already doing the imageTransformation anyway.
    if (surface_model->get_input("")->shape[3] == 3) {
        m_surface_preprocessor = new dl::image::ImagePreprocessor(surface_model, {123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    } else if (surface_model->get_input("")->shape[3] == 1) {
        m_surface_preprocessor = new dl::image::ImagePreprocessor(surface_model, {123.675}, {58.395});
    } else {
        ESP_LOGE("SURFACE", "Unsupported number of channels: %d", surface_model->get_input("")->shape[3]);
        delete surface_model;
        surface_model = nullptr;
        return false;
    }

    if (!m_surface_preprocessor) {
        ESP_LOGE("SURFACE", "Failed to create image preprocessor");
        delete surface_model;
        surface_model = nullptr;
        return false;
    }
    // surface_model->profile_module();

    return true;
}

bool convert_surface_image(const dl::image::img_t* input_img, dl::image::img_t &output_img, dl::image::pix_type_t target_pix_type) {
    // original height and width
    int orig_height = input_img->height;
    int orig_width = input_img->width;

    // crop to square
    int x_min = 32;
    int x_max = x_min + 48;
    int y_min = 0;
    int y_max = y_min + 192;
    std::vector<int> crop_area = {x_min, y_min, x_max, y_max};

    output_img.height = y_max-y_min;
    output_img.width = x_max-x_min;
    output_img.pix_type = target_pix_type;
    if (target_pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB888)
        output_img.data = malloc(output_img.height * output_img.width * 3); // RGB: 3 bytes per pixel
    else if (target_pix_type == dl::image::DL_IMAGE_PIX_TYPE_GRAY)
        output_img.data = malloc(output_img.height  * output_img.width); // GRAY: 1 byte per pixel

    if (!output_img.data) {
        ESP_LOGE("SURFACE", "Memory allocation failed");
        free(output_img.data);
        return false;
    }

    // Convert using ESP-DL
    surfaceTransformer
        .set_src_img(*input_img)
        .set_src_img_crop_area({x_min, y_min, x_max, y_max})
        .set_dst_img(output_img);

    esp_err_t err = surfaceTransformer.transform<false>();
    if (err != ESP_OK) {
        ESP_LOGE("SURFACE", "Image transformation failed: %d", err);
        free(output_img.data);
        return false;
    }

    return true;
}

std::vector<dl::cls::result_t> run_surface_inference(const dl::image::img_t &input_img) {
    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();
    
    m_surface_preprocessor->preprocess(input_img);

    surface_model->run(); //dl::RUNTIME_MODE_MULTI_CORE);
    const int check = 5;
    SurfacePostProcessor m_postprocessor(surface_model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    ESP_LOGI("SURFACE", "inference in %8.0f us.\n", delta);

    for (auto &res : results) {
        ESP_LOGI("SURFACE", "category: %s, score: %f\n", res.cat_name, res.score);
    }

    return results;
}

bool process_surface_image(const dl::image::img_t* input_img) {
    const auto results = run_surface_inference(*input_img);

    float scores[5] = {0};
    // Map category names to their index in the scores array
    for (const auto& res : results) {
        if (strcmp(res.cat_name, "asphalt") == 0) {
            scores[0] = res.score;
        } else if (strcmp(res.cat_name, "paving_stones") == 0) {
            scores[2] = res.score;
        } else if (strcmp(res.cat_name, "sett") == 0) {
            scores[3] = res.score;
        } else if (strcmp(res.cat_name, "unpaved") == 0) {
            scores[4] = res.score;
        }
        // scores[1] remains 0 for 'compacted' placeholder
    }

    notify_surface_classification(scores);
    
    return true;
}