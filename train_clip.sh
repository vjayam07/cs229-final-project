export sample_dataset_name="sample_data/global_street_view_images"
export sample_metadata="sample_data/global_street_view_metadata.csv"

export dataset_name="full_data/global_street_view_images/home/kapil32703/cs229-final-project/global_street_view_images/"
export metadata="full_data/street_view_metadata.csv"

huggingface-cli login

python3 train_countryclip.py \
    --dataset_name=$dataset_name \
    --metadata_file=$metadata \
    --output_dir=""