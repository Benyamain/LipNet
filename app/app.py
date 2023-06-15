import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("LipNet")
    st.info("Focuses on decoding text from a speaker's mouth movements. By using deep learning techniques and advanced models, the project achieves impressive accuracy in mapping video frames to text. With its end-to-end training approach and utilization of spatiotemporal convolutions and recurrent networks, LipNet outperforms previous methods and even surpasses human lip readers in sentence-level accuracy.")
