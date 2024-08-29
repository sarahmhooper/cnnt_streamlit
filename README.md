# CNNT Streamlit
Streamlit User Interface using CNNT for Microscopy Denoising

Run inference on new data or finetune a model on collected paired data

## Instructions

Create a conda env:
```
conda env create -f env.yml
```

Activate the env:
```
conda activate st_env
```

Run the command:
```
streamlit run main.py --server.maxUploadSize 51200 -- --model_path_dir ./saved_models/ --cuda_devices cuda
```

Your default browser should open with the streamlit user interface ready to go!

Provides two pretrained model:
- ./saved_models/cnnt_Microscopy...pth : Backbone model trained with 3DRCAN iSIM denoising data
- ./saved_models/Tile_test_AV_1.pth : Finetuned model for confocal data
