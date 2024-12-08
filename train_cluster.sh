export sample_dataset_name="sample_data/global_street_view_images"
export sample_metadata="sample_data/global_street_view_metadata.csv"

export dataset_name="full_data/global_street_view_images/home/kapil32703/cs229-final-project/global_street_view_images/"
export metadata="full_data/cluster_metadata.csv"
export cluster_centers_metadata=""full_data/cluster_centers.csv"

# huggingface-cli login

python3 train_classifiers.py \
    --dataset_name=$dataset_name \
    --metadata_file=$metadata \
    --cluster_centers=$cluster_centers_metadata \
    --output_dir=""