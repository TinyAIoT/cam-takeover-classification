#ifndef DCI_CALCULATOR_H
#define DCI_CALCULATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "imu.h"

#ifdef __cplusplus
extern "C" {
#endif

// DCI configuration
#define DCI_GRAVITY_THRESHOLD   1.0f                // 1g threshold
#define DCI_MIN_SAMPLES         3                   // Minimum samples for valid DCI (arbitrary, can be tuned)

// DCI result structure
typedef struct {
    float dci_value;           // Calculated DCI
    int total_samples;         // Total acceleration samples processed
    int above_threshold;       // Number of samples > gravity threshold
    float gravity_reference;   // Calculated gravity reference (1g value)
    bool valid;               // Whether DCI calculation is valid
} dci_result_t;

// Functions
esp_err_t dci_calculate_gravity_reference(const imu_data_t* imu_samples, int count, float* gravity_ref);
esp_err_t dci_calculate(const imu_data_t* imu_samples, int count, float gravity_ref, dci_result_t* result);
esp_err_t dci_calculate_with_auto_gravity(const imu_data_t* imu_samples, int count, dci_result_t* result);
#ifdef __cplusplus
}
#endif

#endif /* DCI_CALCULATOR_H */