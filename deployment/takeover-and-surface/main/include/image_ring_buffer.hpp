#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "dl_image_define.hpp"
#include "esp_camera.h"
#include "esp_log.h"

#define RING_BUFFER_SIZE 16

class ImageRingBuffer {
public:
    ImageRingBuffer();
    ~ImageRingBuffer();

    bool add_image(const dl::image::img_t& img);

    bool is_full();

    int get_count();

    dl::image::img_t* get_image(int index);

    bool compose_4x4_image(dl::image::img_t& output_img);

private:
    std::vector<std::vector<uint8_t>> buffer_;
    dl::image::img_t images_[RING_BUFFER_SIZE];
    int write_index_;
    int count_;
    mutable std::mutex mutex_;
};