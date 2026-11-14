// XIAO ESP32-S3 Sense: AprilTag test camera.
//
// Captures exactly like the drone (main/at_detect.c in CDE1302_UAV_Group_3):
// 320x240 grayscale, 2 PSRAM buffers, same sensor tweaks. Frames go to the
// laptop raw (no JPEG), so the laptop detector sees the same pixels the drone
// would. The laptop can change exposure/gain live with text commands.
//
// Frame:  "FRM!" u16 width, u16 height, u32 len, <len bytes gray>, "END!"
// Text:   lines starting with "STATUS " or "MSG " between frames.
// Commands (one per line): aec 0|1, exp <0-1200>, agc 0|1, gain <0-30>,
//                          bright <-2..2>, ae <-2..2>, drone, status

#include <Arduino.h>
#include "esp_camera.h"

// XIAO ESP32-S3 Sense camera pins (same as the drone)
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

static bool camera_ok = false;

// What we last asked the sensor for (the driver's own status is not always
// updated by every sensor, so we keep our own copy to report).
static int st_aec = 1, st_exp = 300, st_agc = 1, st_gain = 0, st_bright = 1, st_ae = 0;

static void send_status() {
  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("\nSTATUS pid=0x%x aec=%d exp=%d agc=%d gain=%d bright=%d ae=%d\n",
                s ? s->id.PID : 0, st_aec, st_exp, st_agc, st_gain, st_bright, st_ae);
}

// Same tweaks as init_camera() in the drone's at_detect.c
static void apply_drone_defaults() {
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) s->set_vflip(s, 1);
  s->set_brightness(s, 1);
  s->set_saturation(s, -2);
  s->set_exposure_ctrl(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_ae_level(s, 0);
  st_aec = 1; st_agc = 1; st_bright = 1; st_ae = 0;
}

static void handle_command(String line) {
  line.trim();
  if (line.isEmpty()) return;
  sensor_t *s = esp_camera_sensor_get();
  int sp = line.indexOf(' ');
  String cmd = sp < 0 ? line : line.substring(0, sp);
  int v = sp < 0 ? 0 : line.substring(sp + 1).toInt();

  if (cmd == "aec")         { st_aec = v ? 1 : 0; s->set_exposure_ctrl(s, st_aec);
                              if (!st_aec) s->set_aec_value(s, st_exp); }
  else if (cmd == "exp")    { st_exp = constrain(v, 0, 1200); s->set_aec_value(s, st_exp); }
  else if (cmd == "agc")    { st_agc = v ? 1 : 0; s->set_gain_ctrl(s, st_agc);
                              if (!st_agc) s->set_agc_gain(s, st_gain); }
  else if (cmd == "gain")   { st_gain = constrain(v, 0, 30); s->set_agc_gain(s, st_gain); }
  else if (cmd == "bright") { st_bright = constrain(v, -2, 2); s->set_brightness(s, st_bright); }
  else if (cmd == "ae")     { st_ae = constrain(v, -2, 2); s->set_ae_level(s, st_ae); }
  else if (cmd == "drone")  { apply_drone_defaults(); }
  else if (cmd != "status") { Serial.printf("\nMSG unknown command: %s\n", line.c_str()); }
  send_status();
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.begin(115200);
  Serial.setTxTimeoutMs(500);

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_GRAYSCALE;  // same as drone
  c.frame_size = FRAMESIZE_QVGA;         // 320x240, same as drone
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&c) != ESP_OK) {
    Serial.println("MSG camera init failed");
    return;
  }
  apply_drone_defaults();
  esp_camera_sensor_get()->set_pixformat(esp_camera_sensor_get(), PIXFORMAT_GRAYSCALE);
  camera_ok = true;
  send_status();
}

static String cmd_buf;

void loop() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n') { handle_command(cmd_buf); cmd_buf = ""; }
    else if (cmd_buf.length() < 64) cmd_buf += ch;
  }
  if (!camera_ok) {
    Serial.println("MSG camera not initialised");
    delay(1000);
    return;
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(5); return; }
  uint16_t w = fb->width, h = fb->height;
  uint32_t len = fb->len;
  Serial.write((const uint8_t *)"FRM!", 4);
  Serial.write((const uint8_t *)&w, 2);
  Serial.write((const uint8_t *)&h, 2);
  Serial.write((const uint8_t *)&len, 4);
  Serial.write(fb->buf, len);
  Serial.write((const uint8_t *)"END!", 4);
  esp_camera_fb_return(fb);
}
