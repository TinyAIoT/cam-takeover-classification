# Surface and Takeover classification
![system design](../../figures/system_design.png)

The camera captures an image and then triggers surface and takeover classification in parallel. Current framerate is ~2.5Hz. Running the two classification tasks in parallel is somehow a little slower than running them simultaneously. The model inference doubles in parallel mode (250ms) compared to simultaneous mode (110ms).