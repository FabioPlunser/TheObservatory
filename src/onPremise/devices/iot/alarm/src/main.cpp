#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <ESP8266mDNS.h>
#include <ESP8266HTTPClient.h>
#include <LittleFS.h>

// WiFi credentials
const char *ssid = "PBS_66829D_TP-Link";
const char *password = "06645306763";

// Piezo buzzer pin
const int BUZZER_PIN = D1;

// WebSocket details
String wsHost;
String wsPath;
String alarm_id;
int wsPort = 0;
bool alarm_active = false;

// WebSocket client
WebSocketsClient webSocket;

String generateUUID()
{
  uint32_t chipId = ESP.getChipId();
  char uuid[37];
  snprintf(uuid, sizeof(uuid),
           "%08x-%04x-%04x-%04x-%04x%08x",
           chipId,
           random(0xFFFF), random(0xFFFF),
           random(0xFFFF), random(0xFFFF),
           ESP.getFlashChipId());
  return String(uuid);
}

String readAlarmID()
{
  if (!LittleFS.begin())
  {
    Serial.println("Failed to mount file system");
    return "";
  }

  if (LittleFS.exists("/alarm_id.txt"))
  {
    File file = LittleFS.open("/alarm_id.txt", "r");
    if (!file)
    {
      Serial.println("Failed to open alarm_id file");
      return "";
    }
    String id = file.readString();
    file.close();
    Serial.println("Loaded existing alarm ID: " + id);
    return id;
  }
  return "";
}

void saveAlarmID(String id)
{
  if (!LittleFS.begin())
  {
    Serial.println("Failed to mount file system");
    return;
  }

  File file = LittleFS.open("/alarm_id.txt", "w");
  if (!file)
  {
    Serial.println("Failed to open alarm_id file for writing");
    return;
  }
  file.print(id);
  file.close();
  Serial.println("Saved alarm ID: " + id);
}

bool discoverServer()
{
  Serial.println("Looking for EdgeServer...");

  if (!MDNS.begin("esp8266alarm"))
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

void registerAlarm()
{
  WiFiClient client;
  HTTPClient http;

  DynamicJsonDocument doc(1024);
  doc["alarm_id"] = alarm_id;
  doc["name"] = "ESP8266-Alarm";
  doc["type"] = "piezo";

  String json;
  serializeJson(doc, json);

  String url = "http://" + wsHost + ":" + String(wsPort) + "/register/alarm";
  http.begin(client, url.c_str());
  http.addHeader("Content-Type", "application/json");
  int httpResponseCode = http.POST(json);

  if (httpResponseCode > 0)
  {
    Serial.printf("Alarm registered successfully, response: %d\n", httpResponseCode);
  }
  else
  {
    Serial.printf("Registration failed, error: %d\n", httpResponseCode);
  }

  http.end();
}

void handleAlarm()
{
  if (alarm_active)
  {
    tone(BUZZER_PIN, 1000); // 1kHz tone
    delay(500);
    noTone(BUZZER_PIN);
    delay(500);
  }
  else
  {
    noTone(BUZZER_PIN);
  }
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
  switch (type)
  {
  case WStype_DISCONNECTED:
    Serial.println("WebSocket disconnected");
    alarm_active = false; // Safety: disable alarm when disconnected
    break;

  case WStype_CONNECTED:
    Serial.println("WebSocket connected");
    break;

  case WStype_TEXT:
  {
    String message = String((char *)payload);
    DynamicJsonDocument doc(200);
    DeserializationError error = deserializeJson(doc, message);

    if (!error)
    {
      if (doc.containsKey("active"))
      {
        alarm_active = (doc["active"] == "true");
        Serial.printf("Alarm state changed to: %s\n", alarm_active ? "ON" : "OFF");
      }
    }
    break;
  }

  case WStype_ERROR:
    Serial.println("WebSocket error");
    break;
  }
}

void setup()
{
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  noTone(BUZZER_PIN);

  // Generate alarm ID
  alarm_id = readAlarmID();
  if (alarm_id.length() == 0)
  {
    alarm_id = generateUUID();
    saveAlarmID(alarm_id);
  }

  Serial.println("Alarm UUID: " + alarm_id);

  // Important: Match the server's WebSocket path format
  wsPath = String("/ws/alarms/") + alarm_id; // Changed from /ws/alarm/ to /ws/alarms/
  Serial.println("WebSocket path: " + wsPath);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // Discover edge server
  while (!discoverServer())
  {
    Serial.println("Retrying server discovery in 5 seconds...");
    delay(5000);
  }

  // Register alarm with server
  registerAlarm();

  // Setup WebSocket connection
  webSocket.begin(wsHost.c_str(), wsPort, wsPath.c_str());
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
}

void loop()
{
  webSocket.loop();
  MDNS.update();
  handleAlarm();
}