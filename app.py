import streamlit as st
from pages.multiapp import MultiApp
from pages import home, cam, image # import your app modules here


app = MultiApp()
st.set_page_config(page_title="FaceBot",page_icon='🤖',layout="wide")
hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden; }
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #708090;">
    <div style="
    font-family: 'Pacifico', cursive;
    font-size: 30px;
   ">Face bot</div>
  </nav>
""", unsafe_allow_html=True)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Webcam", cam.app)
app.add_app("Image", image.app)
# The main app
app.run()
