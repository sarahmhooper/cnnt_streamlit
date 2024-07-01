


import os
import json
import torch
from argparse import Namespace
from model.models import CNNT_enhanced_denoising_runtime


models_dir = "./saved_models"


for name in os.listdir(models_dir):

    if not name.endswith(".pt"):
        continue

    print(name)

    model_pt_path = os.path.join(models_dir, name)
    model_js_path = os.path.join(models_dir, f"{name[:-3]}.json")

    status = torch.load(model_pt_path)
    config = json.load(open(model_js_path))
    config = Namespace(**config)
    config.load_path = None
    # model = CNNT_enhanced_denoising_runtime(config=config)



    torch.save({
        "epoch":config.num_epochs,
        "model_state": status,
        "config": config,
    }, f"{model_pt_path}h")

    
    print("done", name)



