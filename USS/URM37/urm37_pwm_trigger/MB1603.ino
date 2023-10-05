
// # Connection:
// #       Vcc (Arduino)    -> Pin 6 VCC (URM V5.0)
// #       GND (Arduino)    -> Pin 7 GND (URM V5.0)
// #       Pin 3 (Arduino)  -> Pin 2 ECHO (URM V5.0)
// #       Pin 5 (Arduino)  -> Pin 4 COMP/TRIG (URM V5.0)



int URECHO = 3;         // PWM Output 0-50000US,Every 50US represent 1cm
int URTRIG = 5;         // trigger pin

unsigned int DistanceMeasured = 0;

void setup()
{
  //Serial initialization
  Serial.begin(9600);                        // Sets the baud rate to 9600
  // pinMode(URTRIG, OUTPUT);                   // A low pull on pin COMP/TRIG
  // digitalWrite(URTRIG, HIGH);                // Set to HIGH
  pinMode(URECHO, INPUT);                    // Sending Enable PWM mode command
  delay(500);
  Serial.println("Init the sensor");

}
void loop()
{
  Serial.print("Distance=");
  // digitalWrite(URTRIG, LOW);
  // delay(5); // short delay that sensor detects trigger signal
  // digitalWrite(URTRIG, HIGH);              

  unsigned long LowLevelTime = pulseIn(URECHO, HIGH) ;
  if (LowLevelTime > 5000)              // the reading is invalid.
  {
    Serial.println("Invalid");
  }
  else
  {
    DistanceMeasured = LowLevelTime ;  // every 1us low level stands for 1mm
    Serial.print(DistanceMeasured);
    Serial.print("cm     ");

    Serial.print("Time=");
    Serial.print(LowLevelTime);
    Serial.println("us");
  }

  delay(10);
}