#include "image_ring_buffer.hpp"

ImageRingBuffer::ImageRingBuffer() {
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        images_[i].data     = nullptr;
        images_[i].height   = 0;
        images_[i].width    = 0;
        images_[i].pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
    }
    write_index_ = 0;
    count_ = 0;
}

ImageRingBuffer::~ImageRingBuffer() {
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        if (images_[i].data) {
            free(images_[i].data);
            images_[i].data = nullptr;
        }
    }
}

bool ImageRingBuffer::add_image(const dl::image::img_t& img) {
    // Free existing data at this position if any
    if (images_[write_index_].data) {
        free(images_[write_index_].data);
    }
    
    // Allocate new memory and copy image data
    size_t data_size = img.height * img.width * 3; // RGB888: 3 bytes per pixel
    // TODO: fix memory leak
    // if (images_[write_index].data) {
    //     free(images_[write_index].data);
    //     images_[write_index].data = nullptr;
    // }
    images_[write_index_].data = malloc(data_size);
    if (!images_[write_index_].data) {
        ESP_LOGE("TAKEOVER", "Memory allocation failed for ring buffer");
        return false;
    }
    
    memcpy(images_[write_index_].data, img.data, data_size);
    images_[write_index_].height = img.height;
    images_[write_index_].width = img.width;
    images_[write_index_].pix_type = img.pix_type;

    write_index_ = (write_index_ + 1) % RING_BUFFER_SIZE;
    if (count_ < RING_BUFFER_SIZE) {
        count_++;
    }
    
    return true;
}

bool ImageRingBuffer::is_full() { 
    return count_ == RING_BUFFER_SIZE; 
}

int ImageRingBuffer::get_count() { 
    return count_; 
}

dl::image::img_t* ImageRingBuffer::get_image(int index) {
    if (index >= count_) {
        return nullptr;
    }
    int actual_index = (write_index_ - count_ + index + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    return &images_[actual_index];
}

bool ImageRingBuffer::compose_4x4_image(dl::image::img_t& output_img) {
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