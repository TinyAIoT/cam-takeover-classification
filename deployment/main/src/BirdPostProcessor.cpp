#include "BirdPostProcessor.hpp"
#include "bird_category_name.hpp"
#include "dl_model_base.hpp"

BirdPostProcessor::BirdPostProcessor(
    dl::Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name) :
    ClsPostprocessor(model, top_k, score_thr, need_softmax, output_name)
{
    m_cat_names = bird_cat_names;
}
