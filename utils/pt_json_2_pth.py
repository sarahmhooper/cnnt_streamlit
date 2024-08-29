"""
File to convert the older version model saving to the new one

.pt + .json to .pth

Point it to a directory and it will convert them all
"""

import os
import sys
import json
import torch
from argparse import Namespace

models_dir = sys.argv[1]

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

    torch.save({
        "epoch":config.num_epochs,
        "model_state": status,
        "config": config,
    }, f"{model_pt_path}h")

    print("done", name)
