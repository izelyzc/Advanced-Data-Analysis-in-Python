# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 00:37:34 2022

@author: izely

"""

import gc 
gc.collect()

# Import Libraries
from Kmeans_Clustering import data_prep

## Cluster Count
cluster_behav_k=[3,5,7]
cluster_value_k=[2,4,6]

## Call functions
df_stand,df_ADS_clean=data_prep(df_ADS_all) ## Data Prep
df_value_seg,df_behav_seg,df_value_elbow_o,df_behav_elbow_o=clustering(df_stand,cluster_value_k,cluster_behav_k) ## Clustering Outputs & Elbow Analysis
df_value_seg,df_behav_seg=predict_cluster(df_stand,cluster_behav_k,cluster_value_k)   ## Predict Clusters 

# def main():
#     data_prep(df_ADS_all)
#     clustering(df_stand,cluster_value_k,cluster_behav_k)
    
        
# if __name__== '__main__': 
#     main()