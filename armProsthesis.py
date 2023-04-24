import serial
import time

ser = serial.Serial(port = "/dev/ttyACM0", baudrate = 9600)

def sendData(rps):
    if rps == 1:
        data = [0,0,0,0,0]
    elif rps == 2:
        data = [0,1,1,0,0]
    elif rps == 3:
        data = [1,1,1,1,1]
    
    dataString = "$"

    for i in data:
        dataString += str(int(i)).zfill(1)
    try:
        ser.write(dataString.encode())
        return True
    except:
        return False

rps = 1
sendData(rps)
time.sleep(2)
rps = 3
sendData(rps)
time.sleep(10)
    

classifierOutput = [2, 2, 1, 3, 3, 2, 2, 2, 1, 1, 3, 1, 3, 2, 3, 3, 2, 2]
# 1: rock
# 2: scissor
# 3: paper

for i in range(0,len(classifierOutput)):
    rps = classifierOutput[i]
    sendData(rps)
    time.sleep(1.5)
