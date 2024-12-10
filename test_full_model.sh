export sample_dataset_name="sample_data/global_street_view_images"
export sample_metadata="sample_data/global_street_view_metadata.csv"

export dataset_name="full_data/home/kapil32703/cs229-final-project/global_street_view_images/"
export metadata="full_data/street_view_metadata.csv"
# export dataset_name="full_data/test_global_street_view_images"
# export metadata="full_data/test_global_street_view_metadata.csv"

export cluster_centers_metadata="full_data/cluster_centers.csv"

export clip_dir="vjayam07/geoguessr-clip-model"

huggingface-cli login

python3 cs229_geoguessr_prediction.py \
    --dataset_name=$dataset_name \
    --metadata_file=$metadata \
    --cluster_centers=$cluster_centers_metadata \
    --clip_dir=$clip_dir 
