import socket
import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.mouse import Listener
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep
# Define the client class
class Client:
    def __init__(self, host='localhost', port=80):
        self.host = host
        self.port = port
        self.close_socket = False
        self.cont = 0
        
    def start(self):
        # Start the client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        print("Connected to the server.")
        try:
            print("[INFO] FRAME SELECTED")
            img = cv2.imread('/workspace/Pruebas/image.png')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', 400, 400)
            cv2.imshow('screen',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            result = pickle.dumps(img)
            print(f"[INFO] Sending Data...")
            client_socket.sendall(result)
            client_socket.sendall(str.encode("foto"))
            sleep(1)
            data = client_socket.recv(5)
            # decode to unicode string 
            people_counter = int(data.decode('utf8'))
            print("[INFO] Data Received...")
            print(f"[INFO] People: {people_counter}")
            print("*"*80)
    
        finally:
            client_socket.close()
            print("Connection closed.")

# Start the client
if __name__ == "__main__":
    client = Client()
    client.start()
