/*
  The circuit:
 * LCD RS pin to digital pin 7
 * LCD Enable pin to digital pin 8
 * LCD D4 pin to digital pin 9
 * LCD D5 pin to digital pin 10
 * LCD D6 pin to digital pin 11
 * LCD D7 pin to digital pin 12
 * LCD R/W pin to ground
 * LCD VSS pin to ground
 * LCD VCC pin to 5V
 * 10K resistor:
 * ends to +5V and ground
 * wiper to LCD VO pin (pin 3)
 */

#include <LiquidCrystal.h>
#include <Arduino.h>

// Initialize the LCD library with the numbers of the interface pins
LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

int analogPin = A0;
int outputPin = 2;

int inputValue  = 0;

String inputString = "";
bool stringComplete = false;

void sendStatus();

void setup()
{
    // Set up the LCD's number of columns and rows:
    lcd.begin(16, 2);
    Serial.begin(9600);
    while(!Serial)
    {
        ; // wait for serial port to connect. Needed for native USB port only
    }
}

void loop()
{
    if (stringComplete)
    {
        if (inputString.startsWith("status"))
        {
            sendStatus();
        }
        else if (inputString.startsWith("set"))
        {
            if (inputString.indexOf("on") > -1)
            {
                lcd.display();
            }
            else if (inputString.indexOf("off") > -1)
            {
                lcd.clear();
                lcd.noDisplay();
            }
            else
            {
                lcd.clear();
                lcd.print("Unknown set command");
            }
        }
        else if(inputString.startsWith("clear"))
        {
            lcd.clear();
        }
        else
        {
            lcd.clear();
            lcd.print("Unknown command");
        }

        stringComplete = false;
        inputString = "";
    }

    delay(10);
}

void sendStatus()
{
    char buffer[200];
    inputValue = analogRead(analogPin);
    sprintf(buffer, "Analog input %d is %d", analogPin, inputValue);
    lcd.print(buffer);
}

void serialEvent()
{
    while (Serial.available())
    {
        char inChar = (char)Serial.read();
        inputString += inChar;
        if (inChar == '\n')
        {
            stringComplete = true;
        }
    }
}
