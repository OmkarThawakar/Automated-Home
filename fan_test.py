import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)


relay = [0,2,3,27,17]

GPIO.setup(relay[1],GPIO.OUT)
GPIO.setup(relay[2],GPIO.OUT)
GPIO.setup(relay[3],GPIO.OUT)
GPIO.setup(relay[4],GPIO.OUT)

GPIO.output(relay[1],GPIO.HIGH)
GPIO.output(relay[2],GPIO.HIGH)
GPIO.output(relay[3],GPIO.HIGH)
GPIO.output(relay[4],GPIO.HIGH)

def set_relay(num):
  for i in range(1,5):
    if i==num :
      GPIO.output(relay[i],GPIO.LOW)
    else:
      GPIO.output(relay[i],GPIO.HIGH)
  

def restore_relay():
  GPIO.output(relay[1],GPIO.HIGH)
  GPIO.output(relay[2],GPIO.HIGH)
  GPIO.output(relay[3],GPIO.HIGH)
  GPIO.output(relay[4],GPIO.HIGH)


try :
  while 1:
    tempfile = open('/sys/bus/w1/devices/28-04165732b9ff/w1_slave')
    thetext = tempfile.read()
    tempfile.close()
            
    tempdata = thetext.split('\n')[1].split(' ')[9]
    temperature = float(tempdata[2:])
    temperature = temperature / 1000

            
    print('Temperature : ',temperature)

    if 20 < temperature < 26 :
      restore_relay()
      set_relay(3)
      
    elif 25 < temperature < 31 :
      restore_relay()
      set_relay(2)

    elif 30 < temperature < 36 :
      restore_relay()
      set_relay(1)

    else:
      set_relay(4)

    time.sleep(1)

except KeyboardInterrupt:
  GPIO.cleanup()





