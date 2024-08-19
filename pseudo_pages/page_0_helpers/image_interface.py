
import math
import streamlit as st

from utils.utils import *

from inputs.input_class import Input_Class
from model.model_class import Model_Class
from outputs.output_class import Output_Class

ic : Input_Class = st.session_state.input_class
mc : Model_Class = st.session_state.model_class
oc : Output_Class = st.session_state.output_class

def image_reader_st():

    is_inf = is_inf_mode()
    
    noisy_images = st.file_uploader("Noisy Image(s)", accept_multiple_files=True)

    # clean_img_str = "Clean Image(s) (optional for comparison)" if is_inf else "Clean Image(s)"
    # clean_images = st.file_uploader(clean_img_str, accept_multiple_files=True)
    clean_images = []

    # if noisy_images == [] or clean_images == []:
    #     st.stop()

    if len(noisy_images):

        format_a, format_d = ic.read_inputs_files(noisy_images, clean_images)

        display_image_info()

        # format_a, format_d = get_formats(format_a, format_d)
        ic.set_format(format_a=format_a, format_d=format_d)



def display_image_info():

    def build_info_table():

        noisy_name_dict = {}
        clean_name_dict = {}
        shape_dict = {}

        for i in range(ic.get_num_images()):
            noisy_name_dict[f"{i}"] = ic.get_noisy_im_name(i)
            clean_name_dict[f"{i}"] = ic.get_clean_im_name(i)
            shape_dict[f"{i}"] = ic.get_noisy_im_shape(i)

        final_dict = {
            "Noisy":noisy_name_dict,
            "Clean":clean_name_dict,
            "Shape":shape_dict,
        }

        return final_dict

    # # 5 images per tab
    # num_images = ic.get_num_images()
    # images_per_tab = 5
    # num_tabs = math.ceil(num_images/images_per_tab)

    # tab_list = st.tabs([f"Images {i*5}-{i*images_per_tab+4}" for i in range(num_tabs)])

    # for i, tab in enumerate(tab_list):

    #     start = i*5
    #     end = min(num_images, i*5+5)

    #     with tab:
    #         st.table(build_info_table(start, end))

    st.dataframe(build_info_table(), use_container_width=True)

    return

