# cnnt_streamlit
Streamlit UI for CNNT

Run inference on new data or finetune a model on collected paired data

streamlit run main.py --server.maxUploadSize 51200 -- -D --model_path_dir ./saved_models/ --cuda_devices 0,1,2,3
