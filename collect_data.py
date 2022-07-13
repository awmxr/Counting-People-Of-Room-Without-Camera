import socket
import datetime as dt
import os
from time import sleep

# host1 = "172.20.10.5"
host1 = "192.168.43.92"
# host2 = "172.20.10.3"
host2 = "192.168.43.81"
# # host3 = "172.20.10.2"
host3 = "192.168.43.93"
print("waiting 10s ...")
sleep(2)
port = 8888


start = dt.datetime.now()

f = open("data/3/01.txt" , "w")
j = 0
start = dt.datetime.now()
while True:
    client1 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server1 = (host1,port)

    client1.connect(server1)
    # print("connect1")

    client2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server2 = (host2,port)

    client2.connect(server2)
    # print("connect2")
    client3 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server3 = (host3,port)

    client3.connect(server3)

    led1 = client1.recv(3).decode()
    led2 = client2.recv(3).decode()
    led3 = client3.recv(3).decode()

    client1.close()
    client2.close()
    client3.close()


    list01 = led1.split("-")
    for i in list01:
        if len(i) == 2:
            y1 = "-" + i

    
    list02 = led2.split("-")
    for i in list02:
        if len(i) == 2:
            y2 = "-" + i
    
    
    list03 = led3.split("-")
    for i in list03:
        if len(i) == 2:
            y3 = "-" + i
    
    if len(y1) == 3 and len(y2) == 3 and len(y3) == 3:
    
        write_str = y1 + " " + y2 + " " + y3 + "\n"
        f.write(write_str)
        
    
    else:
        continue

    
    os.system("cls")
    print("x1 : " ,y1)
    print("x2 : " ,y2)
    print("x3 : " ,y3)
    j += 1
    if (dt.datetime.now() - start > dt.timedelta(seconds=300) ):
        break
    

f.close()
print(j)
timef = dt.datetime.now() - start
print(timef)
    