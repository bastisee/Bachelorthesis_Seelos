#include <Arduino.h>
#include <ADC.h>

ADC *adc = new ADC();

const int mic1Pin = A0; // ADC0
const int mic2Pin = A1; // ADC1
const int mic3Pin = A2; // ADC0 again

const int SAMPLE_RATE = 44100;
const int TIMER_INTERVAL_US = 1000000 / SAMPLE_RATE;

const int BUFFER_SIZE = 512;
const int PRE_IMPACT = 200;
const int POST_IMPACT = 312;
const int THRESHOLD = 2000; // Increased threshold to reduce false triggers
const int COOLDOWN_MS = 2000; // Minimum time between impacts in milliseconds

int16_t mic1_buf[BUFFER_SIZE];
int16_t mic2_buf[BUFFER_SIZE];
int16_t mic3_buf[BUFFER_SIZE];
int bufferIndex = 0;

bool impactDetected = false;
int postImpactCounter = 0;
int impactCounter = 1;
elapsedMillis cooldown; 

IntervalTimer timer;

void setupADC() {
  adc->adc0->setAveraging(0);
  adc->adc0->setResolution(12);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

  adc->adc1->setAveraging(0);
  adc->adc1->setResolution(12);
  adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
}

void sampleMics() {
  adc->adc0->startSingleRead(mic1Pin);
  adc->adc1->startSingleRead(mic2Pin);
  while (adc->adc0->isConverting() || adc->adc1->isConverting());

  int16_t m1 = adc->adc0->readSingle();
  int16_t m2 = adc->adc1->readSingle();
  int16_t m3 = adc->adc0->analogRead(mic3Pin);

  mic1_buf[bufferIndex] = m1;
  mic2_buf[bufferIndex] = m2;
  mic3_buf[bufferIndex] = m3;

  static int16_t baseline = 2048;
  int amplitude1 = abs(m1 - baseline);
  int amplitude2 = abs(m2 - baseline);
  int amplitude3 = abs(m3 - baseline);
  
  if (!impactDetected && cooldown > COOLDOWN_MS &&
      (amplitude1 > THRESHOLD || amplitude2 > THRESHOLD || amplitude3 > THRESHOLD)) {
      impactDetected = true;
      postImpactCounter = POST_IMPACT;
      cooldown = 0;
  }
  

  if (impactDetected) {
    postImpactCounter--;
    if (postImpactCounter <= 0) {
      Serial.print("# Impact ");
      Serial.println(impactCounter++);

      int start = bufferIndex - PRE_IMPACT;
      if (start < 0) start += BUFFER_SIZE;

      for (int i = 0; i < PRE_IMPACT + POST_IMPACT; i++) {
        int idx = (start + i) % BUFFER_SIZE;
        Serial.print(mic1_buf[idx]); Serial.print(",");
        Serial.print(mic2_buf[idx]); Serial.print(",");
        Serial.println(mic3_buf[idx]);

        
      }
      Serial.println("# End");
      impactDetected = false;
    }
  }

  bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
}

void setup() {
  Serial.begin(500000);
  delay(1000);

  setupADC();
  timer.begin(sampleMics, TIMER_INTERVAL_US);
}

void loop() {
  
}
