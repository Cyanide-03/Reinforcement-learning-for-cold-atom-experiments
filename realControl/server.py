import zmq
import subprocess
import cv2
import base64
import numpy as np
import json

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("ARTIQ server running")

def blank_image():
    img = np.zeros((50, 50), dtype=np.uint8)
    return img

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode()

while True:

    msg = socket.recv_json()
    detuning = msg["detuning"]

    subprocess.run([
        "artiq_run",
        "demo.py"
    ])

    # load images produced by experiment
    # imgs = [
    #     cv2.imread("img1.png")
    # ]

    imgs = [blank_image()]

    encoded_imgs = [encode_image(img) for img in imgs]

    socket.send_json({
        "images": encoded_imgs
    })