#pragma once
#include "dl_cls_postprocessor.hpp"
#include "dl_model_base.hpp"

class ClassificationPostProcessor : public dl::cls::ClsPostprocessor {
public:
    ClassificationPostProcessor(dl::Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name = "");
};
