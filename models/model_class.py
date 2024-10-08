"""
File for the model class

Hold items related to the model
"""
import os

from utils.utils import *
from models.model_variations import load_model, update_config, model_list_from_dir

# -------------------------------------------------------------------------------------------------

class Model_Class():
    """
    Class to hold the model

    Provides ways to setup and load model
    """

    def __init__(self, args):
        # save path_dir given when starting the streamlit server

        self.model = None
        self.config = None
        self.model_name = None
        self.model_path = None

        self.device = args.device
        self.model_path_dir = args.model_path_dir

    def get_model_list(self):
        # Retrieve the possible models from given model path directory

        return model_list_from_dir(self.model_path_dir)

    def load_model(self, model_name, model_file=None):
        # load a model from name or from uploaded model file

        self.model_path = os.path.join(self.model_path_dir, model_name)
        if model_file is None:
            self.model, self.config = load_model(model_path=self.model_path, device=self.device)
        else:
            self.model, self.config = load_model(model_file=model_file, device=self.device)
        self.model_name = model_name

    def update_config(self, config_update):

        self.config = update_config(self.config, config_update)

    def reload_model(self):
    
        self.model, self.config = load_model(config=self.config)

    def is_model_loaded(self):
        # TODO: bug: sometimes unsyncs with whether model is truly loaded or not

        return self.model is not None
