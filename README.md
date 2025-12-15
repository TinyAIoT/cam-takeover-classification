# cam-takeover-classification
Detecting dangerous takover maneuvers live with a microcontroller on a bicycle. But now with a camera!

## proposed system design (software)
![system design](figures/system_design.png)

- pytorch for model design, training and evaluation
- deployment in ESP-IDF

## proposed system design (hardware)
<img src="figures/prototype.png" alt="prototype" style="width:49%"/>
<img src="figures/bike_setup.png" alt="bike_setup" style="width:49%"/>

### ideas and future work
- calculate precision and recall per takeover GROUP (if one frame of a group is takeover, then the whole group is takeover)
