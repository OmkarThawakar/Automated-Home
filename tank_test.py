import RPi.GPIO as GPIO
import time

channel_1 = 20
channel_2 = 21
pin= 16 # for switch

GPIO.setmode(GPIO.BCM)

GPIO.setup(channel_1,GPIO.IN)
GPIO.setup(channel_2,GPIO.IN)
GPIO.setup(pin, GPIO.OUT) 


def fill_tank():
  GPIO.output(pin,GPIO.HIGH)

start_time = time.time()
while True:
  try:
    print(GPIO.input(channel_1) , GPIO.input(channel_2))
    if GPIO.input(channel_1)==1 and GPIO.input(channel_2)==1 :
      print('Tank Full!!!!!')
      GPIO.output(pin,GPIO.LOW)
      GPIO.cleanup()
      print('Time required for filling tank is : ',time.time()-start_time)
    elif GPIO.input(channel_1)==1 and GPIO.input(channel_2)==0 :
      print('Tank Filling in process!!!!')
      fill_tank()
    elif GPIO.input(channel_1)==0 and GPIO.input(channel_2)==0 :
      print('Tank Empty!!!!!')
      fill_tank()
    else:
      print('')

    time.sleep(1)
  except KeyboardInterrupt as e:
    print('Quit')
    GPIO.cleanup()
  

      





