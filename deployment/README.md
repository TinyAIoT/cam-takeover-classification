# some scripts for deploying on the senseBox Eye
The main working project folder is `takover-and-surface-semi-concurrent`. "semi-concurrent" refers to the fact that the classification has its own looping task seperated from the camera capture and distance measurement. But overtake and surface classification are running sequentially.

The project folder `takeover-and-surface` was the initial implementation using a different thread for the camera+distance, the surface classification and the overtake classification. This implementation and its included models are probably not up to date...

`for-testing` can be used to test inference using images from the sd-card or captured live from the camera.

`for-testing-many-models` was used for testing inference times of several models.