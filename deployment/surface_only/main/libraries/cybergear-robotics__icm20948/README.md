# ICM-20948 component for ESP-IDF

[![Component Registry](https://components.espressif.com/components/cybergear-robotics/icm20948/badge.svg)](https://components.espressif.com/components/cybergear-robotics/icm20948)
[![Examples build](https://github.com/cybergear-robotics/icm20948/actions/workflows/build_example.yml/badge.svg)](https://github.com/cybergear-robotics/icm20948/actions/workflows/build_example.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Framework](https://img.shields.io/badge/Framework-ESP-orange.svg)](https://shields.io/)
[![Language](https://img.shields.io/badge/Language-C-purple.svg)](https://shields.io/)

This is a modified copy of the SparkFun Arduino library for the TDK InvenSense ICM-20948 Inertial Measurement Unit 9-Degree Of Freedom sensor from [SparkFun_ICM-20948_ArduinoLibrary](https://github.com/sparkfun/SparkFun_ICM-20948_ArduinoLibrary). It bases on the Portable C99 code and uses the I2C and SPI driver of ESP-IDF. The code bases on Version 1.3 of the SparkFun Arduino library and  includes support for the InvenSense Digital Motion Processor (DMP™). You can find further details in [DMP.md](docs/DMP.md).

## Supported Features
* I2C
* SPI
* DMP

## Using component
```bash
idf.py add-dependency "cybergear-robotics/icm20948"
```

## Example
Not all examples are ported. For further examples, please look at the original project.

1. create example project
```bash
idf.py create-project-from-example "cybergear-robotics/icm20948:i2c_agmt"
```
2. Go to to example directory (for example `spi_agmt`)
   `cd spi_agmt`
3. Set ESP chip
   `idf.py set-target esp32`
4. Configure ICM-20948 settings
   `idf.py menuconfig` and go to `ICM-20948 Example`
5. Build, flash
   `idf.py build flash monitor`

## DMP Support

DMP support can be enabled in menuconfig "ICM-20948 Settings". An example is provided in `examples/spi_dmp_quad9_orientation`.

## Related projects

| Project | Framework | I2C | SPI | DMP |
| :---:   | :---: | :---: | :---: | :---: |
| [SparkFun_ICM-20948_ArduinoLibrary](https://github.com/sparkfun/SparkFun_ICM-20948_ArduinoLibrary) | Arduino | &check;| &check;| &check;|
| [hellvesper/icm20948-i2c](https://github.com/hellvesper/icm20948-i2c) | ESP-IDF | &check;|  | &check;|
| [wollewald/ICM20948_WE](https://github.com/wollewald/ICM20948_WE) | Arduino | &check;| &check;| |
| [isouriadakis/Arduino_ICM20948_DMP_Full-Function](https://github.com/isouriadakis/Arduino_ICM20948_DMP_Full-Function) | Arduino | &check;| &check;| &check;|
