const int RPWM_Pin = 9;  
const int LPWM_Pin = 10; 
const int R_IS_Pin = A0;

int velocidadPWM = 0;

void setup() {
  pinMode(RPWM_Pin, OUTPUT);
  pinMode(LPWM_Pin, OUTPUT);
  Serial.begin(115200);
  // Reducimos el tiempo de espera para que no haya lags
  Serial.setTimeout(20); 
}

void loop() {
  // Lógica blindada: Leemos la línea completa como un String
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); 
    input.trim(); // Quitamos espacios invisibles
    
    if (input.length() > 0) {
      int lectura = input.toInt();
      // Solo actualizamos si es un valor de PWM válido
      if (lectura >= 0 && lectura <= 255) {
        velocidadPWM = lectura;
        analogWrite(RPWM_Pin, velocidadPWM);
        analogWrite(LPWM_Pin, 0);
      }
    }
  }

  // Envío de datos cada 500ms (0.5s)
  static unsigned long lastMillis = 0;
  if (millis() - lastMillis >= 500) {
    lastMillis = millis();
    Serial.print(millis());
    Serial.print(",");
    Serial.print(velocidadPWM);
    Serial.print(",");
    Serial.println(leerEsfuerzo(), 3);
  }
}

float leerEsfuerzo() {
  long suma = 0;
  for(int i = 0; i < 50; i++) {
    suma += analogRead(R_IS_Pin);
    delayMicroseconds(100);
  }
  return (float)suma / 50.0 * (5.0 / 1024.0);
}