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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "BLEModule.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
static uint8_t addr_type;

static const char *manuf_name = "senseBox Eye";
static const char *model_num = "1.1";
uint16_t takeover_classification_hrm_handle;
uint16_t distance_hrm_handle;

static const char *device_name = "senseBox:bike";
static bool notify_state;

static uint16_t conn_handle;


static int gatt_svr_chr_access_takeover_classification(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg);

static int gatt_svr_chr_access_distance(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg);

int gap_event(struct ble_gap_event *event, void *arg)
{
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        /* A new connection was established or a connection attempt failed */
        MODLOG_DFLT(INFO, "connection %s; status=%d\n",
                    event->connect.status == 0 ? "established" : "failed",
                    event->connect.status);

        if (event->connect.status != 0) {
            /* Connection failed; resume advertising */
            advertise();
        }
        conn_handle = event->connect.conn_handle;
        break;

    case BLE_GAP_EVENT_DISCONNECT:
        MODLOG_DFLT(INFO, "disconnect; reason=%d\n", event->disconnect.reason);

        /* Connection terminated; resume advertising */
        advertise();
        break;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        MODLOG_DFLT(INFO, "adv complete\n");
        advertise();
        break;

    case BLE_GAP_EVENT_SUBSCRIBE:
        // TODO: distance??
        MODLOG_DFLT(INFO, "subscribe event; cur_notify=%d\n value handle; "
                    "val_handle=%d\n",
                    event->subscribe.cur_notify, takeover_classification_hrm_handle);
        if (event->subscribe.attr_handle == takeover_classification_hrm_handle) {
            notify_state = event->subscribe.cur_notify;
        } else if (event->subscribe.attr_handle != takeover_classification_hrm_handle) {
            notify_state = event->subscribe.cur_notify;
        }
        ESP_LOGI("BLE_GAP_SUBSCRIBE_EVENT", "conn_handle from subscribe=%d", conn_handle);
        break;

    case BLE_GAP_EVENT_MTU:
        MODLOG_DFLT(INFO, "mtu update event; conn_handle=%d mtu=%d\n",
                    event->mtu.conn_handle,
                    event->mtu.value);
        break;

    }

    return 0;
}

void print_addr(const void *addr)
{
    const uint8_t *u8p;

    u8p = addr;
    MODLOG_DFLT(INFO, "%02x:%02x:%02x:%02x:%02x:%02x",
                u8p[5], u8p[4], u8p[3], u8p[2], u8p[1], u8p[0]);
}

void on_sync(void)
{
    int rc;

    rc = ble_hs_id_infer_auto(0, &addr_type);
    assert(rc == 0);

    uint8_t addr_val[6] = {0};
    rc = ble_hs_id_copy_addr(addr_type, addr_val, NULL);

    MODLOG_DFLT(INFO, "Device Address: ");
    print_addr(addr_val);
    MODLOG_DFLT(INFO, "\n");

    /* Begin advertising */
    advertise();
}

void on_reset(int reason)
{
    MODLOG_DFLT(ERROR, "Resetting state; reason=%d\n", reason);
}

void encodeFloatArrayAsFloatBytes(const float* in, size_t n, uint8_t* out) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        memcpy(&u, &(in[i]), 4);        /* safely grab the bits */

        /* write bytes explicitly as little-endian */
        out[4*i + 0] = (uint8_t)(u & 0xFFu);
        out[4*i + 1] = (uint8_t)((u >> 8)  & 0xFFu);
        out[4*i + 2] = (uint8_t)((u >> 16) & 0xFFu);
        out[4*i + 3] = (uint8_t)((u >> 24) & 0xFFu);
    }
}

// TODO: only when subscribed
void notify_takeover_classification(float values[1])
{
    int rc;

    if (notify_state) {
        // convert to IEEE-754
        uint8_t out[1 * 4];
        encodeFloatArrayAsFloatBytes(values, 1, out);

        struct os_mbuf *om = ble_hs_mbuf_from_flat(out, 1*4);
        if (om == NULL) {
            MODLOG_DFLT(ERROR, "error allocating mbuf for notification\n");
            return;
        }

        rc = ble_gattc_notify_custom(conn_handle, takeover_classification_hrm_handle, om);
        if (rc != 0) {
            MODLOG_DFLT(ERROR, "error sending takeover classificationnotification; rc=%d\n", rc);
        }
    }
}

void notify_distance(float values[1])
{
    int rc;

    if (notify_state) {
        // convert to IEEE-754
        uint8_t out[1 * 4];
        encodeFloatArrayAsFloatBytes(values, 1, out);

        struct os_mbuf *om = ble_hs_mbuf_from_flat(out, 1*4);
        if (om == NULL) {
            MODLOG_DFLT(ERROR, "error allocating mbuf for notification\n");
            return;
        }

        rc = ble_gattc_notify_custom(conn_handle, distance_hrm_handle, om);
        if (rc != 0) {
            MODLOG_DFLT(ERROR, "error sending distance notification; rc=%d\n", rc);
        }
    }
}

/*
 * Enables advertising with parameters:
 *     o General discoverable mode
 *     o Undirected connectable mode
 */
void advertise(void)
{
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;
    int rc;

    /*
     *  Set the advertisement data included in our advertisements:
     *     o Flags (indicates advertisement type and other general info)
     *     o Advertising tx power
     *     o Device name
     */
    memset(&fields, 0, sizeof(fields));

    /*
     * Advertise two flags:
     *      o Discoverability in forthcoming advertisement (general)
     *      o BLE-only (BR/EDR unsupported)
     */
    fields.flags = BLE_HS_ADV_F_DISC_GEN |
                   BLE_HS_ADV_F_BREDR_UNSUP;

    fields.name = (uint8_t *)device_name;
    fields.name_len = strlen(device_name);
    fields.name_is_complete = 1;

    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        MODLOG_DFLT(ERROR, "error setting advertisement data; rc=%d\n", rc);
        return;
    }

    /* Begin advertising */
    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(addr_type, NULL, BLE_HS_FOREVER,
                           &adv_params, gap_event, NULL);
    if (rc != 0) {
        MODLOG_DFLT(ERROR, "error enabling advertisement; rc=%d\n", rc);
        return;
    }
}

void host_task(void *param)
{
    // ESP_LOGI(tag, "BLE Host Task Started");
    /* This function will return only when nimble_port_stop() is executed */
    nimble_port_run();

    nimble_port_freertos_deinit();
}

                            
static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {
        /* Service: Heart-rate */
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID128_DECLARE(GATT_SENSORS_UUID),
        .characteristics = (struct ble_gatt_chr_def[])
        { {
                /* Characteristic: takeover classification */
                .uuid = BLE_UUID128_DECLARE(GATT_TAKEOVER_CLASSIFICATION_UUID),
                .access_cb = gatt_svr_chr_access_takeover_classification,
                .val_handle = &takeover_classification_hrm_handle,
                .flags = BLE_GATT_CHR_F_NOTIFY,
            }, {
                /* Characteristic: distance */
                .uuid = BLE_UUID128_DECLARE(GATT_DISTANCE_UUID),
                .access_cb = gatt_svr_chr_access_distance,
                .val_handle = &distance_hrm_handle,
                .flags = BLE_GATT_CHR_F_NOTIFY,
            }, {
                0, /* No more characteristics in this service */
            },
        }
    },

    {
        0, /* No more services */
    },
};

// TODO: what is this for?
static int
gatt_svr_chr_access_takeover_classification(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    uint16_t uuid;

    uuid = ble_uuid_u16(ctxt->chr->uuid);

    MODLOG_DFLT(INFO, "What do I do with this??");


    assert(0);
    return BLE_ATT_ERR_UNLIKELY;
}

// TODO: what is this for?
static int
gatt_svr_chr_access_distance(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    uint16_t uuid;

    uuid = ble_uuid_u16(ctxt->chr->uuid);

    MODLOG_DFLT(INFO, "What do I do with this??");


    assert(0);
    return BLE_ATT_ERR_UNLIKELY;
}

void
gatt_svr_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg)
{
    char buf[BLE_UUID_STR_LEN];

    switch (ctxt->op) {
    case BLE_GATT_REGISTER_OP_SVC:
        MODLOG_DFLT(DEBUG, "registered service %s with handle=%d\n",
                    ble_uuid_to_str(ctxt->svc.svc_def->uuid, buf),
                    ctxt->svc.handle);
        break;

    case BLE_GATT_REGISTER_OP_CHR:
        MODLOG_DFLT(DEBUG, "registering characteristic %s with "
                    "def_handle=%d val_handle=%d\n",
                    ble_uuid_to_str(ctxt->chr.chr_def->uuid, buf),
                    ctxt->chr.def_handle,
                    ctxt->chr.val_handle);
        break;

    case BLE_GATT_REGISTER_OP_DSC:
        MODLOG_DFLT(DEBUG, "registering descriptor %s with handle=%d\n",
                    ble_uuid_to_str(ctxt->dsc.dsc_def->uuid, buf),
                    ctxt->dsc.handle);
        break;

    default:
        assert(0);
        break;
    }
}

int
gatt_svr_init(void)
{
    int rc;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    rc = ble_gatts_count_cfg(gatt_svr_svcs);
    if (rc != 0) {
        return rc;
    }

    rc = ble_gatts_add_svcs(gatt_svr_svcs);
    if (rc != 0) {
        return rc;
    }

    return 0;
}