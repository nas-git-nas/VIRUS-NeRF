// # Connection:
// #       Vcc (Arduino)    -> Pin 6 VCC (URM V5.0)
// #       GND (Arduino)    -> Pin 7 GND (URM V5.0)
// #       Pin 3 (Arduino)  -> Pin 2 ECHO (URM V5.0)
// #       Pin 5 (Arduino)  -> Pin 4 COMP/TRIG (URM V5.0)



int PIN_PWM = 3;         // PWM Output pin
int PIN_NOISE1 = 5;
int PIN_NOISE2 = 6;
unsigned long DELAY = 100000; // delay in us
bool NOISE = false;

void setup()
{
  //Serial initialization
  Serial.begin(9600);                        // Sets the baud rate to 9600
  pinMode(PIN_PWM, INPUT);
  pinMode(PIN_NOISE1, INPUT);
  pinMode(PIN_NOISE2, INPUT);
  digitalWrite(PIN_NOISE1, LOW);
  digitalWrite(PIN_NOISE2, LOW);
  delay(500);
  Serial.println("Init the sensor");

}

void loop()
{
  unsigned long start_time = micros();

  if(NOISE) {
    digitalWrite(PIN_NOISE1, HIGH);
    digitalWrite(PIN_NOISE2, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_NOISE1, LOW);
    digitalWrite(PIN_NOISE2, LOW);
  }

  unsigned long dist = pulseIn(PIN_PWM, HIGH) ; // every 1us low level stands for 1mm
  Serial.print("meas=");
  Serial.println(dist);

  unsigned long ellapse_time = micros() - start_time;
  if(DELAY > ellapse_time) {
    delayMicroseconds(DELAY - ellapse_time);
  }
  
}
