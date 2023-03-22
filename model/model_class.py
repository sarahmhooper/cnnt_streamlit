"""
File for the model class

Hold items related to the model:
- name and directory
- model itself
"""
import os

from model.models_variations import load_model
from model.models_variations import filter_f
from model.running_inference import running_inference


class Model_Class():
    """
    Model class for background inference model
    Loads the model

    Provides ways to setup and load model
    Provides method to run inference using the model
    """

    def __init__(self, args):
        # only save path_dir given when starting the streamlit server

        self.model = None
        self.config = None
        self.model_path = None

        self.device = args.device
        self.cutout = args.cutout
        self.overlap = args.overlap
        self.model_path_dir = args.model_path_dir

    def get_model_list(self):
        # Retrieve the possible models from given model path directory

        return filter_f(self.model_path_dir)

    def set_model_path(self, model_name, model_files):
        # Given model name, set the model path. Load later
        if model_files == []:
            model_path = os.path.join(self.model_path_dir, model_name)
            self.model_files = []
            self.model_path = model_path
        else:
            self.model_files = model_files
            self.model_path = None

    def load_model(self):
        # Load model before inference
        
        self.model = load_model(model_path=self.model_path, model_files=self.model_files)

    def run_inference(self, cut_np_images):
        # Run inference on loaded model and given images
        # cut_np_images: 3D numpy images of axis order: THW

        clean_pred_list = running_inference(self.model, cut_np_images, self.cutout, self.overlap, self.device)

        return cut_np_images, clean_pred_list
