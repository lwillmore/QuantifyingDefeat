/*
Continuously listen for signals for on or off
*/


#define LED 12

// Using http://slides.justen.eng.br/python-e-arduino as refference

void setup() {
    pinMode(LED, OUTPUT);
    Serial.begin(9600);
}

void loop() {
    if (Serial.available()) {
        char serialListener = Serial.read();
        if (serialListener == '1') {
            // 5 pulses with pulse width of 5 ms
            // frequency is 20 Hz
            for (int i = 0; i < 5; i++) {
              digitalWrite(LED, HIGH);
              delay(5);
              digitalWrite(LED, LOW);
              delay(45);
            }
            Serial.flush();
        }
        else if (serialListener == '0') {
            digitalWrite(LED, LOW);
        }
    }
}