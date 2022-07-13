
#include <ESP8266WiFi.h>
#include <stdlib.h>

const char *ssid = "HP30L";
const char *password = "faradzeropass99";

//const char *ssid = "iPhone";
//const char *password = "amir1380marvast";




int port = 8888;

WiFiServer server(port);







void setup() {
  
  Serial.begin(9600);
  WiFi.begin(ssid,password);
  Serial.println("Connecting to WIFI");

  while (WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
    
    }
    Serial.println(WiFi.localIP());
    Serial.println("");
    Serial.println("Connected to ");
    Serial.println(ssid);
    server.begin();

    
    
  // put your setup code here, to run once:

}

void loop() {
  WiFiClient client = server.available();
  Serial.println("No Client...");
//  int i = 0;
  while (client.connected()){
  //Serial.println("Client Connected but no data send...");
//    int rssi = ;
//    char destination[3] = "S";
    char  str2 [3];
    sprintf(str2, "%d", WiFi.RSSI());
//    strcat(destination,str2);
//    strcat(destination,"F");
//    strcat(destination,"p");
    //char *x = "RSSI: %S dBm\n", str2;
    client.print(str2);
//    Serial.println(str2);
    delay(2);
  
 
  }
//    char destination[15] = "<S>";
//    char  str2 [12];
//    sprintf(str2, "%d", WiFi.RSSI());
//    strcat(destination,str2);  
//    strcat(destination,"<F>");
//    Serial.printf(destination);
  
  

    
//  }

  

}
