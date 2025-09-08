#include "Adafruit_NeoPixel.h"

Adafruit_NeoPixel pixels(1, 45, NEO_GRB + NEO_KHZ800);

void init_LED() {
    pixels.begin();
}

void set_LED(int r, int g, int b, int brightness) {
    pixels.setPixelColor(0, pixels.Color(r, g, b));
    pixels.setBrightness(brightness);
    pixels.show();
    
}