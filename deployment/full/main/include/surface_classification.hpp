#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"
#include "esp_camera.h"

#include "SurfacePostProcessor.hpp"
#include "surface_category_name.hpp"

bool initialize_surface_model();
bool convert_surface_image(const dl::image::img_t* input_img, dl::image::img_t &output_img);
std::vector<dl::cls::result_t> run_surface_inference(const dl::image::img_t &img);
bool process_surface_image(const dl::image::img_t* input_img, float &score, const char** category);