# Surface and Takeover classification
![system design](../../figures/system_design.png)
(currently idea A is implemented)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~3.5Hz. Goal framerate is 5Hz (or more). Running the two classification tasks in parallel is somehow a little slower than running them serially. The model inference doubles in parallel mode (110-270ms) compared to serial mode (110ms). The [docs](https://docs.espressif.com/projects/esp-dl/en/latest/esp-dl-en-master.pdf) mention that conv layers automatically support dual-core, so maybe that has sth to do with the performance loss in the parallel threads.

## TODO

- [x] BLE
    - [X] same transmission format as old senseBox:bike
- [x] measure distance of takeover with ToF
    - [ ] I have to disconnect power to use this. otherwise I get a NACK issue
- [ ] second ToF
- [ ] acceleration sensor?
- [ ] increase framerate
    - smaller models (4bit instead of 8bit)
        - this is prob not going to make it faster... just smaller... but Ill still try
    - [x] classify serially and with multicore -> this made the first classification run much faster!
    - [x] mutex -> didnt help
- [ ] better models
    - more data
    - ideaB for takeover classification
- refactoring
    - [ ] sensor class (+ class for classifications?)
    - [ ] clean up BLE
    - [ ] subfolders (somehow thats not trivial... I have issues with the cmake)
- [x] fix memory leak (overflow after ~30s)
    - [x] fix other memory issue (heap != NULL && "free() target pointer is outside heap areas")
- [x] RGB-LED