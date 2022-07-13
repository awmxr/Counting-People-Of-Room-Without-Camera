import socket

import datetime as dt
import tensorflow as tf
import numpy as np
import os
from termcolor import colored
# import matplotlib.pyplot as plt
STEP = 18*18 # 324
dim = int(np.sqrt(STEP))
NUM_SENSORS = 3
NUM_CLASS = 3 # (0, 1, 2) NAFAR
EPOCHS = 50
TRAIN_PERCENTAGE = 0.8
MODEL_PATH = 'C:/Users/Amir/Desktop/final/model10/'

with open(MODEL_PATH+'min_max_data.txt', 'r') as f:
    min_max_str = f.readline()
min_max_str = min_max_str.split(',')
min_data = float(min_max_str[0])
max_data = float(min_max_str[1])

model = tf.keras.models.load_model(MODEL_PATH)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])





# print("wait 10s")
# time.sleep(3)
# host1 = "172.20.10.5"
host1 = "192.168.43.92"
# host2 = "172.20.10.3"
host2 = "192.168.43.81"
# host3 = "172.20.10.2"
host3 = "192.168.43.93"

port = 8888

# client1 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server1 = (host1,port)

# client1.connect(server1)
# print("connect1")

# client2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server2 = (host2,port)

# client2.connect(server2)
# print("connect2")
# client3 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server3 = (host3,port)

# client3.connect(server3)
# print("connect3")
# rss1_list1 = []
# y_list1 = []
# rss1_list2 = []
# y_list2 = []
# rss1_list3 = []
# y_list3 = []
j = 0

start = dt.datetime.now()
f_list = []
pre = -1
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
    # print("connect3")
        # os.system("cls")
        # if(dt.datetime.now() - start > dt.timedelta(seconds=300)):
        #     break

    
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
            break

    
        
    list02 = led2.split("-")
    for i in list02:
        if len(i) == 2:
            y2 = "-" + i
            break
    
    
    list03 = led3.split("-")
    
    
    for i in list03:
        if len(i) == 2:
            y3 = "-" + i
            break
    
    
    # print("x9")
    
    listxnote = [int(y1) , int(y2) , int(y3)]
    print(listxnote)
    # print("x11")
    # if len(y1) == 3 and len(y2) == 3 and len(y3) == 3:
    
    #     listxnote.append(int(y1))
    #     listxnote.append(int(y2))
    #     listxnote.append(int(y3))
        
    
     
    # else:
    #     # print("x18")
    #     continue

    

    
    

    if len(f_list) == STEP:
        
        # print("x19")
        f_list.pop(0)
        listxnote = np.array(listxnote)
        f_list.append(listxnote)
        listxnote = list(listxnote)
        
        client1.close()
        client2.close()
        client3.close()
        # print(dt.datetime.now() - start)
        # break
        # print(f_list)
        # print(len(f_list))
        # print("xxxxxx")
        # time.sleep(10)
        
        # print("sleepppp")
        # time.sleep(5)
    else:
        # print("x20")
        # print(len(f_list))
        listxnote = np.array(listxnote)
        f_list.append(listxnote)
        listxnote = list(listxnote)
        client1.close()
        client2.close()
        client3.close()
        continue


    
    f_list = np.array(f_list)
    # print(len(f_list))
    # print("xxx")
    # time.sleep(10)
    
    
    f2_list = f_list
    
    # f_list = np.reshape(f_list, (1, STEP, NUM_SENSORS))
    f_list = (((f_list- min_data)/(max_data - min_data)) * 2) - 1

    f_list = np.reshape(f_list, (1, dim, dim, NUM_SENSORS))
    # print("x1")
    prediction_probability = probability_model.predict(f_list)
    # print("x2")
    
    # print('prediction_probability: ', prediction_probability)
    
    prediction = np.argmax(prediction_probability)
    
           
    if pre != prediction:
        # os.system("cls") 
        print(dt.datetime.now())

        if prediction == 0:
            print(y1)
            print(y2)
            print(y3)
            print(colored("\n------------------------------------------\n" , "red"))
            print('\t\tClass: ', prediction )
            print(colored("\n------------------------------------------\n" , "red"))
        elif prediction == 1:
            print(y1)
            print(y2)
            print(y3)
            print(colored("\n------------------------------------------\n" , "blue"))
            print('\t\tClass: ', prediction )
            print(colored("\n------------------------------------------\n" , "blue"))
        elif prediction == 2:
            print(y1)
            print(y2)
            print(y3)
            print(colored("\n------------------------------------------\n" , "green"))
            print('\t\tClass: ', prediction )
            print(colored("\n------------------------------------------\n" , "green"))
            
            pre = prediction
    
    
    # j += 1
    # print(j)
    # print("x1 : " ,x1)
    # print("x2 : " ,x2)
    # print("x2 : " ,x3)
    f_list = f2_list
    f_list = list(f_list)
    # print(f_list)

    # print("x28")
    # time.sleep(10)
    # time.sleep(0.1)
    


    