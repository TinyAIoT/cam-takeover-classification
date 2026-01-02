# Surface and Takeover classification
![system design](../../figures/system_design.png)
(currently idea A is implemented)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~3.5Hz. Goal framerate is 5Hz (or more). Running the two classification tasks in parallel is somehow a little slower than running them serially. The model inference doubles in parallel mode (110-270ms) compared to serial mode (110ms). The [docs](https://docs.espressif.com/projects/esp-dl/en/latest/esp-dl-en-master.pdf) mention that conv layers automatically support dual-core, so maybe that has sth to do with the performance loss in the parallel threads.

In order for the classification task to run indefinitely the watchdog timer has to be disabled in the settings.

Also the ToF will not be initialized correctly if the device is not powered off after a restart.

## TODO

- [ ] second ToF
- [ ] acceleration sensor?
- [ ] better models
    - more data
    - [x] grayscale
- refactoring
    - [ ] sensor class (+ class for classifications?)
    - [ ] clean up BLE
    - [ ] subfolders (somehow thats not trivial... I have issues with the cmake)