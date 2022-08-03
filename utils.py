from pandas import cut
import streamlit as st

def get_cutout():

    time_cutout = st.selectbox(
                "Time cutout",
                (16, 20, 24, 30, 32, "custom"))

    if time_cutout == "custom":
        time_cutout = st.number_input("Time cutout", min_value=0, format="%d")

    HW_cutout = st.selectbox(
                    "Height and Width cutout (square patch)",
                    (64, 128, 192, 256, "custom"))

    if HW_cutout == "custom":
        HW_cutout = st.number_input("Height and Width cutout (square patch)", min_value=0, format="%d")

    cutout_shape = (time_cutout, HW_cutout, HW_cutout)

    st.write("Cutout shape is")
    st.write(cutout_shape)

    
    agree = st.checkbox('Run inference on the above cutout shape?')

    if agree:
        return cutout_shape

    st.stop()