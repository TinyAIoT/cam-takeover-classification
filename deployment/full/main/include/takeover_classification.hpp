#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "esp_camera.h"

#include "TakeoverPostProcessor.hpp"
#include "takeover_category_name.hpp"

bool convert_takeover_image(dl::image::img_t &input_img, dl::image::img_t &output_img);
std::vector<dl::cls::result_t> run_takeover_inference(const dl::image::img_t &img);
bool process_takeover_image(dl::image::img_t &input_img, float &score, const char** category);
