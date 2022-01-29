from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import argparse
from time import sleep
import streamlit as st

st.title("Age Bot")
run = st.checkbox("Run")
FRAME_WINDOW=st.image([])
cam = cv2.VideoCapture(0)

# load model
model = load_model('OLDgender_detection.model')
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(24-32)', '(38-43)', '(48-53)','60-100)']
genderList = ['Man','Woman']
ageNet=cv2.dnn.readNet(ageModel,ageProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding=20
# open webcam
    
while run:
    ret , frame = cam.read()
    # apply face detection  
    face, confidence = cv.detect_face(frame)
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    # loop through detected faces
    for idx, f in enumerate(face):
        face_crop = frame[max(0,f[1]-padding):min(f[3]+padding,frame.shape[0]-1),max(0,f[0]-padding):min(f[2]+padding, frame.shape[1]-1)]
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1] #bottom left
        (endX, endY) = f[2], f[3] #top right
        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2) #giving green color , thickness=2

        # crop the detected face region
        #face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        blob=cv2.dnn.blobFromImage(face_crop, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)            
        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        
        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]
        # get label with max accuracy
        idx = np.argmax(conf)
        label = genderList[idx]      
        if (label == genderList[0]):              # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (255,0,0), 2) #giving green color , thickness=2
        else :
            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (203,192,255), 2) #giving pink color , thickness=2
        label="{}: {:.2f}%,{}".format(label,conf[idx] * 100,age)
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # display output
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.write('stop running')