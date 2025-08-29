# Surface and Takeover classification
![system design](../../figures/system_design.png)
(currently idea B is implemented)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~2.5Hz. Running the two classification tasks in parallel is somehow a little slower than running them simultaneously. The model inference doubles in parallel mode (250ms) compared to simultaneous mode (110ms).

## TODO

- [ ] BLE
- [ ] measure distance of takeover with ToF
- [ ] acceleration sensor?
- [ ] increase framerate
    - smaller models (4bit instead of 8bit)
- [ ] better models
    - more data
    - ideaB for takeover classification