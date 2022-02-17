from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import streamlit as st
from PIL import Image,ImageEnhance

@st.cache
def load_image(img):
    im=Image.open(img)
    return im

eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

def detect_eyes(frame):
    eyes= eye_cascade.detectMultiScale(frame,1.3,5)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame ,eyes

def cartonize_image(frame):
    img = cv2.cvtColor(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Edges
    gray = cv2.medianBlur(gray,5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
    #color
    color = cv2.bilateralFilter(img,9,300,300)
    #Cartoon
    cartoon = cv2.bitwise_and(color,color,mask=edges)
    return cartoon

def object_detection(frame):
    task = ['Eyes','Cartoize']
    feature_choice = st.sidebar.selectbox("Find Feature",task)
    if st.button("Process") :
        if feature_choice =='Eyes':
            res_img ,eyes = detect_eyes(frame)
            st.image(res_img,width=400)
            if eyes is not {}:
                st.success("Found  Eyes")
        elif feature_choice =='Cartoize':
            res_img = cartonize_image(frame)
            st.image(res_img,width=400)

def app():
    st.header("Image bot")
    uploaded_file = st.file_uploader("Choose Your image",type=['jpg','jpeg','png','jfif'])
    model = load_model('models/gender_detection.model')
    ageProto = "models/Age/age_deploy.prototxt"
    ageModel = "models/Age/age_net.caffemodel"
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(24-32)', '(38-43)', '(48-53)','60-100)']
    genderList = ['Man','Woman']
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    padding=20
    
    if uploaded_file is not None:
        our_img = load_image(uploaded_file)
        st.text("Original image :")
        st.image(our_img,width=400)
        Contrast  = st.sidebar.checkbox("Contrast")
        Brightness = st.sidebar.checkbox("Brightness")
        Blurring = st.sidebar.checkbox("Blurring")
        
        img_output=our_img
        
        if Contrast:
            rate = st.sidebar.slider("Contrast",0.5,3.5,1.0)
            enhancer = ImageEnhance.Contrast(img_output)
            img_output = enhancer.enhance(rate)
        if Brightness:
            rate = st.sidebar.slider("Brightness",0.5,3.5,1.0)
            enhancer = ImageEnhance.Brightness(img_output)
            img_output = enhancer.enhance(rate)
        if Blurring :
            img_output = np.array(img_output.convert("RGB"))
            rate = st.sidebar.slider("Blurring",0.5,3.5,1.0)
            img_output = cv2.cvtColor(img_output,1)
            img_output = cv2.GaussianBlur(img_output,(11,11),rate)
            # st.image(img_output,width=400)
        frame =  np.array(img_output)
        face, confidence = cv.detect_face(frame)
        blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
        
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
            object_img = frame
            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            st.write(label)
        st.image(frame,width=400)
        object_detection(frame)
