#include "led.hpp"

// Define the global variables here
led_strip_handle_t strip;

led_strip_config_t cfg = {
    .strip_gpio_num = PIN_LED,
    .max_leds = 1,
    .led_model = LED_MODEL_WS2812,
};

led_strip_rmt_config_t rmt_cfg = {
    .resolution_hz = 10000000,
};

void init_LED() {
    led_strip_new_rmt_device(&cfg, &rmt_cfg, &strip);
    // Clear the LED initially
    led_strip_clear(strip);
}

void set_LED(int r, int g, int b, int brightness) {
    // Scale RGB values by brightness (0-255)
    int scaled_r = (r * brightness) / 255;
    int scaled_g = (g * brightness) / 255;
    int scaled_b = (b * brightness) / 255;
    
    led_strip_set_pixel(strip, 0, scaled_r, scaled_g, scaled_b);
    led_strip_refresh(strip);
}