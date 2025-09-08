# Takeover classification
![system design](../../figures/system_design-only_takeover.png)
(currently idea A is implemented)

The camera captures an image and then triggers only takeover classification in parallel. Current framerate is ~5.8Hz. 

For the vl53l8cx to be correctly initialized it is necessary to turn the device completely off and on again.

## TODO

- [ ] better model
    - more data
    - ideaB for takeover classification
- refactoring
    - [ ] classes and factories
    - [ ] clean up BLE