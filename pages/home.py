import json

import requests  # pip install requests
import streamlit as st
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from PIL import Image

gender = Image.open('style/images/gender.webp')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():
    
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<link rel="stylesheet" href="./style/style.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
        st.markdown("""
            <div class="container" style="padding-top:5%">
                <div class="row">
                    <div class="col">
                        <p style="font-size:40px; font-weight:600; margin-top:100px">Best Face API</p>
                        <p>Confused about someones gender or their age ? Find gender and age from Photo or even from live camera with our application. This app can figure out your gender , age, status, feeling with small amount of editing for fun. </p>
                    </div>
                </div>
            </div>
            <br>
            <div class="container">
            <div class="row">
            <div class="col-md-4">
            <h4>Detect gender</h4>
            <p> Our goal is to find the best result of deep learning gender model With using UTKFace dataset </p>
            </div><div class="col-md-4">
            <h4>Detect age</h4>
            <p>How old do I look? Decide your age from your photographs using our face scanner app </p>
            </div></div>
            </div>
        """,unsafe_allow_html=True)      
    with col2:    
        lottie_bot = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_ofa3xwo7.json")

        st_lottie(
        lottie_bot,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        height="10dp",
        width="5dp",
        key=None,
        )
    