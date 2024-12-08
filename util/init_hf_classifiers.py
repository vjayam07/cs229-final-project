import sys

import pandas as pd
from huggingface_hub import create_repo

def main():
    sys.path.append('../')
    cluster_centers = pd.read_csv("full_data/cluster_centers.csv")
    cluster_counts = cluster_centers.groupby('Country')['Cluster'].nunique()

    countries_multiple_clusters = set(cluster_counts[cluster_counts > 1].index.tolist())
    country_counts = cluster_centers['Country'].value_counts().to_dict()
    
    final_dict = {}
    for country in country_counts:
        if country in countries_multiple_clusters:
            final_dict[country] = country_counts[country]

    print(final_dict)



    # for country in countries_multiple_clusters:
    #     create_repo(f"vjayam07/{country}_classifier", private=True)

if __name__=='__main__':
    main()