export HF_LEROBOT_HOME=/home/gy/Documents/pi05-openpi/data/training 
cd ~/Documents/pi05-openpi

# NOTE: repo_id will be fixed
REPO_ID="arx_a5/put_shrimp_in_pot_openpi"
DATA_DIR="data/arx_a5/put_shrimp_in_pot"

uv run pipeline/align_training_data.py \
    --data_dir ${DATA_DIR} \
    --repo_id ${REPO_ID}

# target data dir: ${HF_LEROBOT_HOME}/${REPO_ID}