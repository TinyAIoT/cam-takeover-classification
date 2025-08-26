# cam-takeover-classification
Detecting dangerous takover maneuvers live with a microcontroller on a bicycle. But now with a camera!

## proposed system design (software)
![system design](figures/system_design.png)

- pytorch for model design, training and evaluation
- deployment in ESP-IDF

### ideas and future work
- data augmentation
- NAS for "custom model" in ideaB
- calculate precision and recall per takeover GROUP (if one frame of a group is takeover, then the whole group is takeover)
