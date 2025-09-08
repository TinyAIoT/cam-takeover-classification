/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef H_BLEHR_SENSOR_
#define H_BLEHR_SENSOR_

#include "nimble/ble.h"
#include "modlog/modlog.h"

#ifdef __cplusplus
extern "C" {
#endif

    #define GATT_SENSORS_UUID                       0x84, 0xBC, 0xB0, 0x1E, 0xBC, 0x8E, 0x04, 0xAD, 0xBE, 0xE0, 0x8E, 0xF6, 0x18, 0xA2, 0x06, 0xCF
    #define GATT_TAKEOVER_CLASSIFICATION_UUID       0x8D, 0xD1, 0xFE, 0xF9, 0x3A, 0x37, 0x18, 0xAE, 0x65, 0x49, 0x44, 0x2C, 0x88, 0xC6, 0x01, 0xFC
    #define GATT_DISTANCE_UUID                      0x2B, 0xA6, 0x37, 0x1F, 0xC9, 0x49, 0x0D, 0xA3, 0x06, 0x43, 0xF3, 0xC0, 0x60, 0x1B, 0x49, 0xB3

    extern uint16_t takeover_classification_hrm_handle;
    extern uint16_t distance_hrm_handle;

    struct ble_hs_cfg;
    struct ble_gatt_register_ctxt;

    void gatt_svr_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg);
    int gatt_svr_init(void);
    void host_task(void *param);
    void advertise(void);
    int gap_event(struct ble_gap_event *event, void *arg);
    void on_sync(void);
    void on_reset(int reason);
    void encodeFloatArrayAsFloatBytes(const float* in, size_t n, uint8_t* out);
    void notify_takeover_classification(float values[1]);
    void notify_distance(float values[1]);

#ifdef __cplusplus
}
#endif

#endif