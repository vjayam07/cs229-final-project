export sample_dataset_name="sample_data/global_street_view_images"
export sample_metadata="sample_data/global_street_view_metadata.csv"

# export dataset_name="full_data/global_street_view_images/home/kapil32703/cs229-final-project/global_street_view_images/"
export dataset_name="full_data/gps_query_imgs"
export metadata="full_data/im2gps_output.csv"
export cluster_centers_metadata="full_data/cluster_centers.csv"

export clip_dir="vjayam07/geoguessr-clip-model"

huggingface-cli login

python3 train_classifiers.py \
    --dataset_name=$dataset_name \
    --metadata_file=$metadata \
    --cluster_centers=$cluster_centers_metadata \
    --clip_dir=$clip_dir
    --output_dir=""