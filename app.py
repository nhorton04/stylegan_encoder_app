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
    try:
        #Upload images
        uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
        if uploaded_file is not None:
            st.image(uploaded_file, width=200)
        second_uploaded_file = st.file_uploader("Choose another picture", type=['jpg', 'png'])
        if second_uploaded_file is not None:
            st.image(second_uploaded_file, width=200)

        img1 = PIL.Image.open(uploaded_file)
        # wpercent = (256/float(img1.size[0]))
        # hsize = int((float(img1.size[1])*float(wpercent)))
        # img1 = img1.resize((256,hsize), PIL.Image.LANCZOS)
        img2 = PIL.Image.open(second_uploaded_file)
        # wpercent = (256/float(img2.size[0]))
        # hsize = int((float(img2.size[1])*float(wpercent)))
        # img2 = img2.resize((256,hsize), PIL.Image.LANCZOS)

        images = [img1, img2]



        # load the StyleGAN model into Colab
        URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
        tflib.init_tf()
        with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
        # load the latents
        s1 = np.load('lrs/176557_10150108810979851_7868129_o_01_01.npy')
        s2 = np.load('lrs/IMG_2482_01_01.npy')
        s1 = np.expand_dims(s1,axis=0)
        s2 = np.expand_dims(s2,axis=0)
        # combine the latents somehow... let's try an average:

        x = st.slider('picture 1', 0.01, 0.9, 0.5)
        y = st.slider('picture 2', 0.01, 0.9, 0.5)
        savg = (x*s1+y*s2)


        # run the generator network to render the latents:
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
        images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)

        for image in images:
            st.image((PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS)))

        # if st.button('Align Images'):
        #     align(images)

        if st.button('Encode Images'):
            main(images)



        for image in images:
            st.image(image, width=200)
    except:
        pass




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
