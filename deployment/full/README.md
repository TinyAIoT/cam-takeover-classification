# Surface and Takeover classification
![system design](../../figures/system_design.png)
(currently idea A is implemented)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~2.7Hz. Goal framerate is 5Hz (or more). Running the two classification tasks in parallel is somehow a little slower than running them serially. The model inference doubles in parallel mode (250ms) compared to serial mode (110ms).

## TODO

- [x] BLE
- [ ] measure distance of takeover with ToF
- [x] fix memory leak (overflow after ~30s)
- [ ] acceleration sensor?
- [ ] increase framerate
    - smaller models (4bit instead of 8bit)
- [ ] better models
    - more data
    - ideaB for takeover classification