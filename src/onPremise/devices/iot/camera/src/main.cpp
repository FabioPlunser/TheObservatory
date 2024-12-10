#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "mbedtls/base64.h"
#include <Preferences.h>
Preferences preferences;

// WiFi credentials
const char *ssid = "PBS_66829D_TP-Link";
const char *password = "06645306763";

// WebSocket server details (will be discovered via mDNS)
String wsHost;
String wsPath;
String camera_id;
String camera_name = "ESP-Cam";
int wsPort = 0;

// Declare WebSocket client once
WebSocketsClient webSocket;

// Update the base64 encoding function to add error handling
String encodeImage(uint8_t *data, size_t len)
{
  if (data == nullptr || len == 0)
  {
    Serial.println("Invalid image data");
    return "";
  }

  size_t encoded_len = (len + 2) / 3 * 4;
  char *encoded = (char *)malloc(encoded_len + 1);

  if (encoded == nullptr)
  {
    Serial.println("Failed to allocate memory for base64");
    return "";
  }

  size_t out_len;
  int result = mbedtls_base64_encode((unsigned char *)encoded, encoded_len + 1, &out_len, data, len);

  if (result != 0)
  {
    Serial.printf("Base64 encoding failed with error: %d\n", result);
    free(encoded);
    return "";
  }

  String base64 = String(encoded);
  free(encoded);

  if (base64.length() == 0)
  {
    Serial.println("Base64 string is empty");
    return "";
  }

  return base64;
}

// Camera pins for AI Thinker ESP32-CAM
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// Generate UUID using ESP32's random number generator
String generateUUID()
{
  uint8_t uuid[16];

  // Fill array with random bytes
  for (int i = 0; i < 16; i++)
  {
    uuid[i] = esp_random() & 0xff;
  }

  // Set version (4) and variant bits
  uuid[6] = (uuid[6] & 0x0f) | 0x40;
  uuid[8] = (uuid[8] & 0x3f) | 0x80;

  // Convert to string
  char uuid_str[37];
  sprintf(uuid_str, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
          uuid[0], uuid[1], uuid[2], uuid[3],
          uuid[4], uuid[5],
          uuid[6], uuid[7],
          uuid[8], uuid[9],
          uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);

  return String(uuid_str);
}

String getCameraId()
{
  preferences.begin("camera", false);
  String id = preferences.getString("uuid", "");
  if (id == "")
  {
    id = generateUUID();
    preferences.putString("uuid", id);
    Serial.println("Generated and saved new camera ID: " + id);
  }
  else
  {
    Serial.println("Loaded camera ID: " + id);
  }

  preferences.end();
  return id;
}

void initCamera()
{
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  // Reduce XCLK frequency for better stability
  config.xclk_freq_hz = 20000000; // Changed from 20MHz to 10MHz

  config.pixel_format = PIXFORMAT_JPEG;

  // Start with lower resolution
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 15; // 0-63 lower means higher quality
  config.fb_count = 2;      // Changed to 2 frame buffers

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Get sensor reference
  sensor_t *s = esp_camera_sensor_get();
  if (s)
  {
    s->set_hmirror(s, 0); // 0 = disable , 1 = enable
    s->set_vflip(s, 0);   // 0 = disable , 1 = enable

    // Initial sensor settings here
    s->set_brightness(s, 0);                 // -2 to 2
    s->set_contrast(s, 0);                   // -2 to 2
    s->set_saturation(s, 0);                 // -2 to 2
    s->set_special_effect(s, 0);             // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
    s->set_whitebal(s, 1);                   // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);                   // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);                    // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
    s->set_exposure_ctrl(s, 1);              // 0 = disable , 1 = enable
    s->set_aec2(s, 0);                       // 0 = disable , 1 = enable
    s->set_gain_ctrl(s, 1);                  // 0 = disable , 1 = enable
    s->set_agc_gain(s, 0);                   // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0); // 0 to 6
    s->set_bpc(s, 0);                        // 0 = disable , 1 = enable
    s->set_wpc(s, 1);                        // 0 = disable , 1 = enable
    s->set_raw_gma(s, 1);                    // 0 = disable , 1 = enable
    s->set_lenc(s, 1);                       // 0 = disable , 1 = enable
    s->set_hmirror(s, 0);                    // 0 = disable , 1 = enable
    s->set_vflip(s, 0);                      // 0 = disable , 1 = enable
    s->set_dcw(s, 1);                        // 0 = disable , 1 = enable
    s->set_colorbar(s, 0);                   // 0 = disable , 1 = enable
  }

  Serial.println("Camera configuration complete");
}

bool discoverServer()
{
  Serial.println("Looking for EdgeServer...");

  if (!MDNS.begin("esp32cam"))
  {
    Serial.println("Error starting mDNS");
    return false;
  }

  int n = MDNS.queryService("edgeserver", "tcp");

  if (n == 0)
  {
    Serial.println("No servers found");
    return false;
  }

  wsHost = MDNS.IP(0).toString();
  wsPort = MDNS.port(0);

  Serial.print("Server discovered at: ");
  Serial.print(wsHost);
  Serial.print(":");
  Serial.println(wsPort);

  return true;
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
  switch (type)
  {
  case WStype_DISCONNECTED:
    Serial.println("WebSocket disconnected");
    break;
  case WStype_CONNECTED:
    Serial.println("WebSocket connected");
    break;
  case WStype_TEXT:
    // Handle incoming messages if needed
    break;
  case WStype_ERROR:
    Serial.println("WebSocket error");
    break;
  }
}

void registerCamera()
{
  WiFiClient client;
  HTTPClient http;

  DynamicJsonDocument doc(1024);
  doc["camera_id"] = camera_id;
  doc["name"] = camera_name;
  doc["capabilities"] = JsonArray();
  doc["capabilities"].add("video");

  String json;
  serializeJson(doc, json);

  String url = "http://" + wsHost + ":" + String(wsPort) + "/register";
  http.begin(client, url.c_str());
  http.addHeader("Content-Type", "application/json");
  int httpResponseCode = http.POST(json);

  if (httpResponseCode > 0)
  {
    Serial.printf("Camera registered successfully, response: %d\n", httpResponseCode);
  }
  else
  {
    Serial.printf("Registration failed, error: %d\n", httpResponseCode);
  }

  http.end();
}

void setup()
{
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  Serial.println();

  // Generate UUID first
  camera_id = generateUUID();
  Serial.println("Camera UUID: " + camera_id);

  // Construct websocket path with UUID
  wsPath = String("/ws/camera/") + camera_id;
  Serial.println("WebSocket path: " + wsPath);

  initCamera();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  while (!discoverServer())
  {
    Serial.println("Retrying server discovery in 5 seconds...");
    delay(5000);
  }

  registerCamera();

  // Use the dynamic wsPath
  webSocket.begin(wsHost.c_str(), wsPort, wsPath.c_str());
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop()
{
  webSocket.loop();

  if (webSocket.isConnected())
  {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("Camera capture failed");
      delay(500); // Wait a bit before trying again
      return;
    }

    if (fb->len == 0)
    {
      Serial.println("Empty frame buffer");
      esp_camera_fb_return(fb);
      delay(500);
      return;
    }

    String encodedImage = encodeImage(fb->buf, fb->len);
    if (encodedImage.length() == 0)
    {
      Serial.println("Failed to encode image");
      esp_camera_fb_return(fb);
      delay(1000);
      return;
    }

    DynamicJsonDocument doc(JSON_OBJECT_SIZE(3) + encodedImage.length() + 1000);
    doc["camera_id"] = camera_id;
    doc["timestamp"] = millis();
    doc["frame"] = encodedImage;

    String jsonString;
    serializeJson(doc, jsonString);

    // Debug print - only print first 100 chars to avoid flooding serial
    // Serial.println("Sending frame (first 100 chars): " + camera_id);

    webSocket.sendTXT(jsonString);

    esp_camera_fb_return(fb);
    delay(100);
  }
}