import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

from align import align
from encode import main

import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
import cv2
import argparse
import numpy as np
import pandas as pd
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
# sys.path.append('tl_gan')
# sys.path.append('pg_gan')
# import feature_axis
# import tfutil
# import tfutil_cpu

# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

def main():

    #Upload images
    uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
    second_uploaded_file = st.file_uploader("Choose another picture", type=['jpg', 'png'])
    if second_uploaded_file is not None:
        st.image(second_uploaded_file, width=200)

    img1 = np.array(uploaded_file)
    img2 = np.array(second_uploaded_file)

    df = pd.Series(img1)
    print(df)


    print(img1)
    print(img2)

    images = [img1, img2]





# @st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
# def load_model():
#     """
#     Create the tensorflow session.
#     """
#     # Open a new TensorFlow session.
#     config = tf.ConfigProto(allow_soft_placement=True)
#     session = tf.Session(config=config)
#
#     # Must have a default TensorFlow session established in order to initialize the GAN.
#     with session.as_default():
#         # Read in either the GPU or the CPU version of the GAN
#         with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, 'rb') as f:
#             G = pickle.load(f)
#     return session, G

if __name__ == "__main__":
    main()
