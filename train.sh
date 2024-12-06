export sample_dataset_name="sample_data/global_street_view_images"
export sample_metadata="sample_data/global_street_view_metadata.csv"

export dataset_name=""
export metadata=""

export OUTPUT_DIR="kanji_model_50k"

python3 train_streetclip.py \
    --dataset_name=$sample_dataset_name \
    --metadata_file=$sample_metadata \
    --output_dir=""



#   --dataset_name=$dataset_name \
#   --use_ema \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --max_train_steps=50000 \
#   --learning_rate=5e-06 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --report_to=wandb \
#   --checkpointing_steps=25000