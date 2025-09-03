# Surface and Takeover classification
![system design](../../figures/system_design.png)
(currently idea A is implemented)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~2.75Hz. Goal framerate is 5Hz (or more). Running the two classification tasks in parallel is somehow a little slower than running them serially. The model inference doubles in parallel mode (270ms) compared to serial mode (110ms).

## TODO

- [x] BLE
    - [X] same transmission format as old senseBox:bike
- [x] measure distance of takeover with ToF
    - [ ] I have to disconnect power to use this. otherwise I get a NACK issue
- [ ] second ToF
- [ ] acceleration sensor?
- [ ] increase framerate
    - smaller models (4bit instead of 8bit)
- [ ] better models
    - more data
    - ideaB for takeover classification
- refactoring
    - [ ] sensor class (+ class for classifications?)
    - [ ] clean up BLE
    - [ ] subfolders (somehow thats not trivial... I have issues with the cmake)
- [x] fix memory leak (overflow after ~30s)
    - [ ] fix other memory issue (heap != NULL && "free() target pointer is outside heap areas")
- [x] RGB-LED