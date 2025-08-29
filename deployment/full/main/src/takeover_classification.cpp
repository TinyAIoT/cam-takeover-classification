#include "takeover_classification.hpp"

// Ring buffer for converted images
#define RING_BUFFER_SIZE 16
struct ImageRingBuffer {
    dl::image::img_t images[RING_BUFFER_SIZE];
    int write_index;
    int count;
    
    ImageRingBuffer() : write_index(0), count(0) {
        // Initialize all images
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            images[i].data = nullptr;
            images[i].height = 0;
            images[i].width = 0;
            images[i].pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
        }
    }
    
    ~ImageRingBuffer() {
        // Free all allocated memory
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            if (images[i].data) {
                free(images[i].data);
                images[i].data = nullptr;
            }
        }
    }
    
    bool add_image(const dl::image::img_t& img) {
        // Free existing data at this position if any
        if (images[write_index].data) {
            free(images[write_index].data);
        }
        
        // Allocate new memory and copy image data
        size_t data_size = img.height * img.width * 3; // RGB888: 3 bytes per pixel
        // TODO: fix memory leak
        // if (images[write_index].data) {
        //     free(images[write_index].data);
        //     images[write_index].data = nullptr;
        // }
        images[write_index].data = malloc(data_size);
        if (!images[write_index].data) {
            ESP_LOGE("TAKEOVER", "Memory allocation failed for ring buffer");
            return false;
        }
        
        memcpy(images[write_index].data, img.data, data_size);
        images[write_index].height = img.height;
        images[write_index].width = img.width;
        images[write_index].pix_type = img.pix_type;
        
        write_index = (write_index + 1) % RING_BUFFER_SIZE;
        if (count < RING_BUFFER_SIZE) {
            count++;
        }
        
        return true;
    }
    
    bool is_full() const {
        return count == RING_BUFFER_SIZE;
    }
    
    int get_count() const {
        return count;
    }
    
    const dl::image::img_t* get_image(int index) const {
        if (index >= count) {
            return nullptr;
        }
        int actual_index = (write_index - count + index + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
        return &images[actual_index];
    }
    
    bool compose_4x4_image(dl::image::img_t& output_img) {
        if (!is_full()) {
            ESP_LOGE("RING_BUFFER", "Cannot compose 4x4 image - buffer not full");
            return false;
        }
        
        // Assuming all images are 24x24 (from convert_takeover_image)
        const int single_img_size = 24;
        const int grid_size = 4;
        const int composed_width = single_img_size * grid_size;  // 96
        const int composed_height = single_img_size * grid_size; // 96
        
        output_img.width = composed_width;
        output_img.height = composed_height;
        output_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
        
        size_t total_size = composed_width * composed_height * 3; // RGB888: 3 bytes per pixel
        // TODO: fix memory leak
        // if (output_img.data) {
        //     free(output_img.data);
        //     output_img.data = nullptr;
        // }
        output_img.data = malloc(total_size);
        if (!output_img.data) {
            ESP_LOGE("RING_BUFFER", "Memory allocation failed for composed image");
            return false;
        }
        
        // Copy each image to its position in the 4x4 grid
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            const dl::image::img_t* src_img = get_image(i);
            if (!src_img || !src_img->data) {
                ESP_LOGE("RING_BUFFER", "Invalid image at index %d", i);
                free(output_img.data);
                return false;
            }
            
            // Calculate grid position (row, col)
            int row = i / grid_size;
            int col = i % grid_size;
            
            // Calculate pixel offsets in the composed image
            int dst_x_start = col * single_img_size;
            int dst_y_start = row * single_img_size;
            
            // Copy pixel by pixel (or row by row for efficiency)
            for (int y = 0; y < single_img_size; y++) {
                for (int x = 0; x < single_img_size; x++) {
                    // Source pixel position
                    int src_offset = (y * single_img_size + x) * 3;
                    
                    // Destination pixel position in composed image
                    int dst_y = dst_y_start + y;
                    int dst_x = dst_x_start + x;
                    int dst_offset = (dst_y * composed_width + dst_x) * 3;
                    
                    // Copy RGB values
                    uint8_t* src_data = (uint8_t*)src_img->data;
                    uint8_t* dst_data = (uint8_t*)output_img.data;
                    
                    dst_data[dst_offset] = src_data[src_offset];         // R
                    dst_data[dst_offset + 1] = src_data[src_offset + 1]; // G
                    dst_data[dst_offset + 2] = src_data[src_offset + 2]; // B
                }
            }
        }
        
        ESP_LOGI("RING_BUFFER", "Composed 4x4 image (%dx%d) from ring buffer", 
                 composed_width, composed_height);
        return true;
    }
};

ImageRingBuffer ring_buffer;
extern const uint8_t espdl_model[] asm("_binary_takeover_espdl_start");
dl::Model *takeover_model = nullptr;
dl::image::ImagePreprocessor *m_takeover_preprocessor = nullptr;

bool initialize_takeover_model() {
    // TODO: but I am not even using an sd-card??
    char dir[64];
    snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    
    takeover_model = new dl::Model((const char *)espdl_model, dir);
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
    // TODO: fix memory leak
    // if (cropped_img.data) {
    //     free(cropped_img.data);
    //     cropped_img.data = nullptr;
    // }
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
    // TODO: fix memory leak
    // if (output_img.data) {
    //     free(output_img.data);
    //     output_img.data = nullptr;
    // }
    output_img.data = malloc(target_h * target_w * 3); // RGB888: 3 bytes per pixel

    // Convert using ESP-DL
    dl::image::resize(cropped_img, output_img, dl::image::DL_IMAGE_INTERPOLATE_BILINEAR);

    free(cropped_img.data);

    return true;
}

const dl::cls::result_t run_takeover_inference(dl::image::img_t &input_img) {    
    uint32_t t0, t1;
    float delta;
    t0 = esp_timer_get_time();
    
    m_takeover_preprocessor->preprocess(input_img);

    takeover_model->run();
    const int check = 5;
    TakeoverPostProcessor m_postprocessor(takeover_model, check, std::numeric_limits<float>::lowest(), true);
    std::vector<dl::cls::result_t> &results = m_postprocessor.postprocess();

    t1 = esp_timer_get_time();
    delta = t1 - t0;
    printf("Inference in %8.0f us.\n", delta);

    dl::cls::result_t best_result = {};
    bool found_result = false;

    for (auto &res : results) {
        ESP_LOGI("TAKEOVER", "category: %s, score: %f\n", res.cat_name, res.score);
        if (!found_result || res.score > best_result.score)
        {
            best_result = res;  // Copy the result
            found_result = true;
        }
    }

    return best_result;
}

bool process_takeover_image(const dl::image::img_t* input_img, float &score, const char** category) {
    dl::image::img_t converted_img;
    if (!convert_takeover_image(input_img, converted_img)) {
        ESP_LOGE("TAKEOVER", "Could not convert image");
        return false;
    }

    // Add converted image to ring buffer
    if (!ring_buffer.add_image(converted_img)) {
        ESP_LOGE("TAKEOVER", "Could not add image to ring buffer");
        free(converted_img.data);
        return false;
    }
    
    // Free the original converted_img.data since it's been copied to ring buffer
    free(converted_img.data);
    converted_img.data = nullptr;
    
    ESP_LOGI("TAKEOVER", "Added image to ring buffer. Count: %d/%d", 
                ring_buffer.get_count(), RING_BUFFER_SIZE);

    // Check if the ring buffer is full
    if (!ring_buffer.is_full()) {
        ESP_LOGI("TAKEOVER", "Ring buffer has not been filled yet.");
        return false; // Not enough images yet
    } else {
        ESP_LOGI("TAKEOVER", "Ring buffer is full. Composing 4x4 image.");
        dl::image::img_t composed_img;
        if (!ring_buffer.compose_4x4_image(composed_img)) {
            ESP_LOGE("TAKEOVER", "Failed to compose 4x4 image");
            return false;
        }

        const auto best = run_takeover_inference(composed_img);
        if (best.cat_name) {
            ESP_LOGI("TAKEOVER", "Best: %s (score: %f)", best.cat_name, best.score);
        }
        
        score = best.score;
        *category = best.cat_name;
    }

    return true;
}
