# Surface and Takeover classification
![system design](../../figures/system_design.png)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is 6.67 Hz. 

## TODO

- [x] BLE
    - [X] same transmission format as old senseBox:bike
- [x] measure distance of takeover with ToF
    - [x] I have to disconnect power to use this. otherwise I get a NACK issue
- [ ] second ToF
- [ ] acceleration sensor?
- [x] increase framerate
    - smaller models 
        - [ ] 4bit instead of 8bit
            - this is prob not going to make it faster... just smaller... but Ill still try
        - [x] grayscale
    - [x] classify serially and with multicore -> this made the first classification run much faster!
    - [x] mutex -> didnt help
- [x] better models
    - more data
    - [x] grayscale
- refactoring
    - [ ] sensor class (+ class for classifications?)
    - [ ] clean up BLE
    - [ ] subfolders (somehow thats not trivial... I have issues with the cmake)
- [x] fix memory leak (overflow after ~30s)
    - [x] fix other memory issue (heap != NULL && "free() target pointer is outside heap areas")
- [x] RGB-LED
