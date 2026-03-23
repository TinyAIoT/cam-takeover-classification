# Surface Type Classification and Roughness Detection Deployment

![system design](../../figures/system_design.png)

Parallel acceleration measurement at 100Hz, and DCI calculation and surface classification at 5 Hz on ESP32-S3.

## System Overview

**Dual-core processing:**
- **Core 0**: Camera capture and surface classification (5 Hz)
- **Core 1**: IMU data collection (20 samples at 100 Hz)

**Hardware:**
- MCU: ESP32-S3
- Camera: OV2640 (QVGA, RGB565)
- IMU: ICM20948 for Dynamic Comfort Index calculation

## Setup Requirements

## Performance

- Camera: 5 Hz capture rate (can be increased if necessary)
- IMU: 100 Hz sampling
- Classification: ~120-130ms latency per frame
- BLE transmission to sensBox:bike app