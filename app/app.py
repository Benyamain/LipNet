import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set layout
st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("LipNet")
    st.info("Focuses on decoding text from a speaker's mouth movements. By using deep learning techniques and advanced models, the project achieves impressive accuracy in mapping video frames to text.")

st.title('LipNet')
# Retrieve the list of videos
videos = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Pick a video', videos)

# Generate 2 columns
col1, col2 = st.columns(2)

if videos:
    with col1:
        st.info('Chosen video:')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
    
    with col2:
        st.text('Column 2')