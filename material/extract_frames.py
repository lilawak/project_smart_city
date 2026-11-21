import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.mouse import Listener

cont = 0
pos = {"x":[],"y":[]}
def is_clicked(x, y, button, pressed):
    global cont, pos
    if pressed:
        print('Clicked ! ') #in your case, you can move it to some other pos
        pos["x"].append(x)
        pos["y"].append(y)
        cont+=1
        if cont == 2:
            return False # to stop the thread after click

with Listener(on_click=is_clicked) as listener:
    listener.join()

bounding_box = {'top': pos["y"][0], 'left': pos["x"][0], 'width': pos["x"][1]-pos["x"][0], 'height': pos["y"][1]-pos["y"][0]}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break