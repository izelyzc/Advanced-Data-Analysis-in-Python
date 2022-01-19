# Import Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from sklearn import model_selection
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pyarrow
    
pd.set_option('display.max_columns',None)

###### Read data from DB
df_GKPI = pd.read_feather("GeneralKPI.feather")
df_GKPI=df_GKPI.drop(columns=['SourceBrandKey','CNT_TRX_L12M_ECOM','CNT_TRX_L24M_ECOM'])
df_SD =  pd.read_csv("SpecialDays.txt", delimiter = "\t")
df_TS =  pd.read_csv("TopStore.txt", delimiter = "|")
df_RA =  pd.read_csv("RewardAnalysis.txt", delimiter = "|")
 

### Join all tables
df_ADS_all= df_GKPI.merge(df_SD, how='left', on=['CustomerKey'])
df_ADS_all= df_ADS_all.merge(df_TS, how='left', on=['CustomerKey'])
df_ADS_all= df_ADS_all.merge(df_RA, how='left', on=['CustomerKey'])

### Calculate segment dimensions
df_ADS_all.loc[df_ADS_all['AMT_L12M']==0,'AMT_L12M']=1

df_ADS_all['GM_L12M_PERC'] = df_ADS_all.apply(lambda x: x.GM_L12M/x.AMT_L12M, axis=1)

# Price_Sensitivity
df_ADS_all.loc[df_ADS_all['TotalPrice_L24M']==0,'TotalPrice_L24M']=1
df_ADS_all['Price_Sensitivity'] = df_ADS_all.apply(lambda x: 1-(x.AMT_L24M/x.TotalPrice_L24M), axis=1)

# Frequency
df_ADS_all['CustomerTenure']=df_ADS_all['CustomerAge'].astype('int')
df_ADS_all.loc[df_ADS_all['CustomerTenure']==0,'CustomerTenure']=1
df_ADS_all['FrequencyMonth'] = np.round(np.where(df_ADS_all['CustomerTenure'] > 24, df_ADS_all['NetTransactionCount_L24M']/24,df_ADS_all['NetTransactionCount_L24M']/df_ADS_all['CustomerTenure'] ), 1)

# DaysToChurn 
df_ADS_all['DaysToChurn'] = df_ADS_all.apply(lambda x: 270-x.DaysSinceLastTransaction_L12M, axis=1)
df_ADS_all.loc[df_ADS_all['DaysToChurn']<= 0, 'DaysToChurn'] = 0
  

def data_prep(data):    
    df_ADS_onebrand=data     
    df_ADS_onebrand_num=df_ADS_onebrand.select_dtypes(['number']) 
    num_col_names=df_ADS_onebrand_num.columns.astype(str).tolist()
    low_lim_list=[]
    up_lim_list=[]
    std_q3_lim=[]
    for i in num_col_names:
        summary=df_ADS_onebrand_num[i].describe()
        Q1=summary[summary.index == '25%'].values
        Q3=summary[summary.index == '75%'].values
        STD=summary[summary.index =='std'].values
        IQR=Q3-Q1
        low_lim = pd.DataFrame(Q1 - (1.5 * IQR)).iloc[0,0]
        up_lim = pd.DataFrame(Q3 + (1.5 * IQR)).iloc[0,0]
        std_q3=pd.DataFrame(Q3 + STD).iloc[0,0]
        low_lim_list.append(low_lim)
        up_lim_list.append(up_lim) 
        std_q3_lim.append(std_q3)
    
    outlier={'col_names':num_col_names,'low_lim':low_lim_list,'up_lim':up_lim_list,'std_q3':std_q3_lim}    
    df_outliers=pd.DataFrame.from_dict(outlier)
    
    df_ADS_onebrand_num_cleaned=df_ADS_onebrand_num.copy()
    
    ## Replace outliers
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['CustomerAge']< 18 ,'CustomerAge']= 18
                   
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['DaysSinceFirstTransaction_L12M']>365 ,'DaysSinceFirstTransaction_L12M']= 365
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['DaysSinceLastTransaction_L12M']>365 ,'DaysSinceLastTransaction_L12M']= 365
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['DaysToChurn']>270 ,'DaysToChurn']= 270

    
    ### Outlier replacement loop for <0 then replace 0
    
    outlier_replacement_lower_limit_zero=['CustomerTenure','FullPriceRatio_L12M','FullPriceRatio_L24M',\
                                          'DaysSinceFirstTransaction_L12M', \
                                          'DaysSinceLastTransaction_L12M','GM_L12M', \
                                          'AMT_L12M','AMT_L24M', \
                                          'NetTransactionCount_L12M' ,'NetTransactionCount_L24M',\
                                          'GrossTransactionCount_L12M', 'GrossTransactionCount_L24M', \
                                          'AVB_L12M','AVB_L24M','TotalPrice_L12M','TotalPrice_L24M', \
                                          'DiscountedPrice_L12M','DiscountedPrice_L24M', \
                                           'Price_Sensitivity', \
                                          'FrequencyMonth','DaysToChurn','GM_L12M_PERC']
        
    for i in outlier_replacement_lower_limit_zero:
        df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned[i]< 0 ,i]= 0
    
    ### Outlier replacement loop for upper limit > 1  then replace 1  
    outlier_replacement_upper_limit_one=['FullPriceRatio_L12M','FullPriceRatio_L24M', \
                                         'GM_L12M_PERC', \
                                         'Price_Sensitivity']
        
    for i in outlier_replacement_upper_limit_one:
        df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned[i]> 1 ,i]= 1
    
    	
    # ### Outlier replacement loop for upper limit > std+q3 value then replace std+q3 value
    # outlier_replacement_upper_limit_std_q3=['CustomerTenure', 'CustomerAge', \
    #                                       'GM_L12M', \
    #                                       'AMT_L12M','AMT_L2M', \
    #                                       'NetTransactionCount_L12M' ,'NetTransactionCount_L24M',\
    #                                       'GrossTransactionCount_L12M', 'GrossTransactionCount_L24M', \
    #                                       'AVB_L12M','AVB_L24M','TotalPrice_L12M','TotalPrice_L24M', \
    #                                       'DiscountedPrice_L12M','DiscountedPrice_L24M']
    # for i in outlier_replacement_upper_limit_std_q3:
    #     std_q3_upper_lim = pd.DataFrame(df_outliers.loc[df_outliers['col_names']== i, 'std_q3']).iloc[0,0] 
    #     df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned[i] > std_q3_upper_lim ,i]= std_q3_upper_lim
    
    
    df_ADS_onebrand_num_cleaned['ReducedPriceRatio_L12M']= 1-df_ADS_onebrand_num_cleaned['FullPriceRatio_L12M']
    df_ADS_onebrand_num_cleaned['ReducedPriceRatio_L24M']= 1-df_ADS_onebrand_num_cleaned['FullPriceRatio_L24M'] 
    
    df_ADS_onebrand_num_cleaned['GM_L24M_PERC_CLV'] = df_ADS_onebrand_num_cleaned['GM_L12M_PERC']
    up_lim_GM= pd.DataFrame(df_outliers.loc[df_outliers['col_names']=='GM_L12M_PERC', 'up_lim']).iloc[0,0]
    low_lim_GM=  pd.DataFrame(df_outliers.loc[df_outliers['col_names']=='GM_L12M_PERC', 'low_lim']).iloc[0,0]
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['GM_L12M_PERC']>up_lim_GM ,'GM_L24M_PERC_CLV']= up_lim_GM
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['GM_L12M_PERC']<low_lim_GM ,'GM_L24M_PERC_CLV']= low_lim_GM
    
    
    # CLV Customer Lifetime Value 
    df_ADS_onebrand_num_cleaned['AnnualCLV'] = df_ADS_onebrand_num_cleaned.apply(lambda x: x.GM_L24M_PERC_CLV * x.AVB_L24M * x.FrequencyMonth, axis=1)
    
    # Round values
    df_ADS_onebrand_num_cleaned[['CustomerTenure',\
                                                             'CustomerAge','DaysSinceFirstTransaction_L12M',\
                                                             'DaysSinceLastTransaction_L12M',\
                                                             'NetTransactionCount_L12M',\
                                                             'NetTransactionCount_L24M',\
                                                             'GrossTransactionCount_L12M',\
                                                             'GrossTransactionCount_L24M',\
                                                             'DaysToChurn']] \
    =df_ADS_onebrand_num_cleaned[['CustomerTenure',\
                                                              'CustomerAge','DaysSinceFirstTransaction_L12M',\
                                                              'DaysSinceLastTransaction_L12M',\
                                                              'NetTransactionCount_L12M',\
                                                              'NetTransactionCount_L24M',\
                                                              'GrossTransactionCount_L12M', 
                                                              'GrossTransactionCount_L24M',\
                                                              'DaysToChurn']].round(0)
  
    #### Imputation
    df_ADS_onebrand_num_cleaned[['DaysSinceLastTransaction_L12M','DaysSinceFirstTransaction_L12M']]=df_ADS_onebrand_num_cleaned[['DaysSinceLastTransaction_L12M','DaysSinceFirstTransaction_L12M']].fillna(value=366)
    df_ADS_onebrand_num_cleaned=df_ADS_onebrand_num_cleaned.fillna(value=0)
    
    
    # Status  
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['DaysToChurn']<= 0,'Status']='Dormant'
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['DaysToChurn']>0,'Status']='Active'
    df_ADS_onebrand_num_cleaned.loc[df_ADS_onebrand_num_cleaned['CustomerTenure']<=4,'Status']='Newcomer'
        
    #### Standardization
    
    from sklearn.preprocessing import MinMaxScaler
    stand_list=['GM_L12M_PERC','AVB_L24M','FrequencyMonth',\
                'FullPriceRatio_L24M']
    column_list=['CustomerKey','Status',\
                  'GM_L12M_PERC','AVB_L24M','FrequencyMonth', \
                'FullPriceRatio_L24M']
    df_stand=df_ADS_onebrand_num_cleaned[column_list]
    
    min_max_scaler =MinMaxScaler()
    for n in stand_list:
        df_stand[[n]]=min_max_scaler.fit_transform(df_stand[[n]])
   
    return(df_stand,df_ADS_onebrand_num_cleaned)


df_stand,df_ADS_clean=data_prep(df_ADS_all) ## Data Prep

############################################ End  of Data Prep Function  ###############################################
def clustering(df_stand,cluster_value_k,cluster_behav_k):
         
    df_stand=df_stand.round(2)
    df_stand_nc=df_stand[df_stand.Status =="Newcomer"]
    df_stand_train=df_stand[df_stand.Status !="Newcomer"]
    ### Behavior Segmentetaion
    # Cluster using SOME columns
    df_behav_seg=df_stand.copy()
    df_behav_seg=df_behav_seg.iloc[0:0]  
    df_behav_elbow_k=pd.DataFrame()     
    for i in cluster_behav_k:
        kmeans_b = KMeans(n_clusters=i, random_state=0).fit(df_stand_train[['FrequencyMonth','FullPriceRatio_L24M']])    
        filename_b='kmeans_behav_model_export_'+'.pickle'
        outfile=open(filename_b,'wb')
        pickle.dump(kmeans_b,outfile)
        outfile.close()
      
        ## Save the labels
        df_stand_train.loc[:,'ClusterId'] = kmeans_b.labels_
        df_stand_train.loc[:,'SSECluster'] = kmeans_b.inertia_
        df_stand_train['SSECluster']  = df_stand_train['SSECluster'].round(0)
        df_stand_train['ClusterVersion']  = i
        df_behav_elbow=df_stand_train.groupby('ClusterVersion')['SSECluster'].mean().round(0).reset_index()
        df_stand_nc['ClusterVersion'] = i
        df_stand_nc['ClusterId'] = -1
        df_behav_seg=pd.concat([df_behav_seg,df_stand_train,df_stand_nc],ignore_index=True)
        df_behav_elbow_k=pd.concat([df_behav_elbow_k,df_behav_elbow],ignore_index=True)
        
        
    ### Value Segmentetaion
    # Cluster using SOME columns
    df_value_seg=df_stand.copy()
    df_value_seg=df_value_seg.iloc[0:0]  
    df_value_elbow_k=pd.DataFrame()   #df_value_seg.copy()
    for i in cluster_value_k:
        kmeans_v = KMeans(n_clusters=i, random_state=0).fit(df_stand_train[['GM_L12M_PERC','AVB_L24M','FrequencyMonth']])
            
        filename_v='kmeans_value_model_export_'+'.pickle'
        outfile=open(filename_v,'wb')
        pickle.dump(kmeans_v,outfile)
        outfile.close()
        # Save the labels
        df_stand_train.loc[:,'ClusterId'] = kmeans_v.labels_
        df_stand_train.loc[:,'SSECluster'] = kmeans_v.inertia_
        df_stand_train['SSECluster']  = df_stand_train['SSECluster'].round(0)
        df_stand_train['ClusterVersion']  = i
        print(i)
        df_stand_nc['ClusterVersion'] = i
        df_value_elbow=df_stand_train.groupby('ClusterVersion')['SSECluster'].mean().round(0).reset_index()
        df_stand_nc['ClusterId'] = -1
        df_value_seg=pd.concat([df_value_seg,df_stand_train,df_stand_nc],ignore_index=True)
        df_value_elbow_k=pd.concat([df_value_elbow_k,df_value_elbow],ignore_index=True)
    return(df_value_seg,df_behav_seg,df_value_elbow_k,df_behav_elbow_k)   

   

def predict_cluster(df_stand,cluster_behav_k,cluster_value_k):
    
    
    import pickle
    
    ### Behaviour Segment Prediction
    df_stand=df_stand.round(2)
    df_stand_nc=df_stand[df_stand.Status =="Newcomer"]
    df_stand_train=df_stand[df_stand.Status !="Newcomer"]
    
    
    filename_b='kmeans_behav_model_export_'+ '.pickle'
    infile_b=open(filename_b,'rb')
    behav_model=pickle.load(infile_b)
    

    pred_behav_clusters=behav_model.fit_predict(df_stand_train[['FrequencyMonth','FullPriceRatio_L24M']])   
    df_stand_train.loc[:,'ClusterId'] = pred_behav_clusters
    df_stand_train['ClusterVersion']  = cluster_behav_k
        
    df_stand_nc['ClusterVersion'] = cluster_behav_k
    df_stand_nc['ClusterId'] = -1
    df_behav_seg=pd.concat([df_stand_train,df_stand_nc],ignore_index=True)
        
    infile_b.close()
    
    ### Value Segment Prediction
    df_stand=df_stand.round(2)
    df_stand_nc=df_stand[df_stand.Status =="Newcomer"]
    df_stand_train=df_stand[df_stand.Status !="Newcomer"]
    
    filename_v='kmeans_value_model_export_' +'.pickle'
    infile_v=open(filename_v,'rb')
    value_model=pickle.load(infile_v)
        
    pred_value_clusters=value_model.fit_predict(df_stand_train[['GM_L12M_PERC','AVB_L24M','FrequencyMonth']])
        
    df_stand_train.loc[:,'ClusterId'] = pred_value_clusters
    df_stand_train['ClusterVersion']  = cluster_value_k
    df_stand_nc['ClusterVersion'] = cluster_value_k
    df_stand_nc['ClusterId'] = -1
    df_value_seg=pd.concat([df_stand_train,df_stand_nc],ignore_index=True)
    
    infile_v.close()
    
    return(df_value_seg,df_behav_seg)

    

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
    
    
    