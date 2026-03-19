#ifndef LED_HPP
#define LED_HPP

#include "led_strip.h"

#define PIN_LED 45

// Declare as extern to avoid multiple definitions
extern led_strip_handle_t strip;
extern led_strip_config_t cfg;
extern led_strip_rmt_config_t rmt_cfg;

void init_LED();
void set_LED(int r, int g, int b, int brightness = 255);

#endif // LED_HPP