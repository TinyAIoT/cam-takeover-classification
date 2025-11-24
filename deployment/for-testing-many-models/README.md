# For Testing a trained and compressed model
This ESP-IDF script is for testing a trained esp-dl model. It can be either tested on images taken by the connected camera or on images on a given sd-card.

There are two variables at the beginning of app_main.cpp which determine if camera or sd-card is used:
```
#define TAKE_PICTURE <false|true>
#define SAVE_TO_SDCARD <false|true>
```

## TAKE_PICTURE true && SAVE_TO_SDCARD false
Images are continuously captured by the camera and classified.

## TAKE_PICTURE true && SAVE_TO_SDCARD true
Images are continuously captured by the camera and classified. The images along with their classification results are saved to sd.

## TAKE_PICTURE false && SAVE_TO_SDCARD true
Images should be given on the sd-card in following folder structure:
- `to_classify`
    - `<classname_A>`
        - ...images...
    - `<classname_B>`
        - ...images...
    - ...

These images are loaded, classified and the result saved to a new folder `classified`. Also a confusion matrix and the absolute accuracy are calculated and saved to a txt on the sd-card.

## TAKE_PICTURE false && SAVE_TO_SDCARD false
Classifies the given `example.jpg` in the `data` folder.

