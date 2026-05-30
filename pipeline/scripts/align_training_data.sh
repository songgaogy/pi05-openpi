export HF_LEROBOT_HOME=/home/gy/Documents/pi05-openpi/data/training 
cd ~/Documents/pi05-openpi

# NOTE: repo_id will be fixed
uv run pipeline/align_training_data.py \
    --data_dir data/arx_a5/put_shrimp_in_pot \
    --repo_id arx_a5/put_shrimp_in_pot_openpi

# target data dir: ${HF_LEROBOT_HOME}/arx_a5/put_shrimp_in_pot_openpi