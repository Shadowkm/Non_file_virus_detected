
# coding: utf-8

# # 從這裡跑到full merge結束（day之前，day不跑）

# In[1]:


from os import listdir
import os
from os.path import isfile, join
import pandas as pd
import numpy as np

# 路徑要修改
mypath="/data/examples/trend/data/query_log"
trainpath="/data/examples/trend/data/training-set.csv"
testpath="/data/examples/trend/data/testing-set.csv"
# train_setpath="train_set.csv"
# test_setpath="test_set.csv"
#mypath="data/examples/trend/query_log" 看一下目錄對不對


# In[2]:


# Data_log=pd.DataFrame()
# # B=np.array(["0301.csv","0302.csv"])
# # for filename in B:

# for filename in listdir(mypath):
#     Data_all=pd.read_csv(os.path.join(mypath,filename),names=["FileID","CustomerID","QueryTs","ProductID"],
#                         low_memory=False)
#     Data_log=pd.concat([Data_log,Data_all],ignore_index=True)


# In[3]:


Data_log=pd.DataFrame()
# B=np.array(["0301.csv","0302.csv"])
# # for filename in B:
# f=pd.concat
# for filename in listdir(mypath):
#     Data_log=f([Data_log,pd.read_csv(os.path.join(mypath,filename),names=["FileID","CustomerID","QueryTs","ProductID"],
#                         low_memory=False)],ignore_index=True)


# In[4]:


Data_log=pd.concat([pd.read_csv(os.path.join(mypath,file_name),header=None) for file_name in listdir(mypath)],ignore_index=True)


# In[293]:


Data_log.columns=["FileID","CustomerID","QueryTs","ProductID"]


# In[294]:


# Data_log.head()


# In[295]:


# for i in [0,1,3]:
#     df[i] = df[i].astype('category')
#     print(df.info())


# In[296]:


Data_log.info()


# In[297]:


for i in ['FileID','CustomerID','ProductID']:
    Data_log[i]=Data_log[i].astype('category')


# In[298]:


Data_log.info()


# In[299]:


# Data_log = Data_log.sort_values(by='QueryTs').reset_index(drop=True)


# In[300]:


# Data_log["CustomerID"].describe()
# 可以討論CustomerID frequency


# In[301]:


# df[2] = pd.to_datetime(df[2], unit='s')


# In[302]:


# Data_log.head()


# In[303]:


time_cal=Data_log["QueryTs"]
ori_time=1488326400 #1488326400 3/1 0:00
week=np.floor((time_cal-ori_time)/(7*24*60*60))
day=np.floor((time_cal-ori_time)/(24*60*60))
hour=np.floor((time_cal-ori_time)/(60*60))
minute=np.floor((time_cal-ori_time)/(60))
ten_second = np.floor((time_cal-ori_time)/(10))
thrity_second = np.floor((time_cal-ori_time)/(30))
six_hour=np.floor((time_cal-ori_time)/(6*60*60))
# 1488326400 為3/1的0:00


# In[304]:


Data_log["Day"]=day
Data_log["Week"]=week
Data_log["Hour"]=hour
Data_log["minute"]=minute
Data_log["ten_second"]=ten_second
Data_log["thirty_second"]=thrity_second
Data_log["six_hour"]=six_hour



# In[305]:


Data_log.head()


# In[306]:


# df=pd.DataFrame(Data_log[Data_log["CustomerID"]=="282396145a3df4452761bacf8049f6db"]).sort_values()


# In[307]:


file_list=[]
for filename in listdir(mypath):
    file_list.append(filename)
print (file_list)
print (len(file_list))


# In[308]:


import time 
from datetime import datetime


# In[309]:


train=pd.read_csv(trainpath,names=["FileID","Detected"])

test=pd.read_csv(testpath,names=["FileID","Detected"])


# In[310]:


# test['Detected']=np.nan


# In[311]:


# 驗證
# Data_log[Data_log.FileID.isin(train.FileID)]


# In[312]:


# # check for missing value
# data_df_na = Data_log.isnull().mean(axis=0)
# data_df_na = data_df_na.drop(data_df_na[data_df_na == 0].index).sort_values(ascending=False)
# missing_data = pd.DataFrame({'Missing Data Ratio': data_df_na})
# print('data_df_na.shape = ', data_df_na.shape)


# In[313]:


import time 
from datetime import datetime

TIME=[]
Day=[]
Hour=[]
unix_time=Data_log["QueryTs"]


# In[314]:


# Train_merge = pd.merge(Data_log, train, how="inner", on='FileID')
# Test_merge = pd.merge(Data_log, test, how="inner", on='FileID')
# full_data


# In[315]:


full_data= pd.concat([train, test])


# In[316]:


# full_data


# In[317]:


full_merge = pd.merge(Data_log, full_data, how="inner", on='FileID')


# In[318]:


full_merge['Detected']=full_merge['Detected'].replace(0.5, 0.5)


# In[319]:


del Data_log,full_data


# In[320]:


full_merge.head()


# In[321]:


from gensim.models import Word2Vec


# In[322]:


# def sum_up(grp):
#     print (len(grp))
#     x=len(grp)
#     return np.diff(x)
# def sum_down(grp):
   
#     return grp[(len(grp<2))]
# def over_200_up(grp):
#     return np.sum(grp)


# In[323]:


# train_diff_D_mean=train_draw.set_index('FileID').groupby(level=0)['Detected'].agg({sum_up})


# In[324]:


full_merge.columns


# In[37]:


# full_merge.groupby(["FileID","Day"])['QueryTs'].count()


# In[38]:


# train_draw.head()


# In[39]:


# train_draw[train_draw['FileID']=='00008c73ee43c15b16c26b26398c1577']['Day']


# # 從這裡開始跟時間有關的都不用跑，已經有另外把相關的feature存起來了，直接跳到“Product ID feature”
# # Day

# In[36]:


train_draw=pd.DataFrame(full_merge.groupby(["FileID","Day"])["Detected"].count()).reset_index()


# In[37]:


train_d_mean=train_draw.set_index('FileID').groupby(level=0)['Detected'].agg({np.mean,np.min,np.max,np.std}).reset_index()
train_d_cumsum=train_draw.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
train_d_mean.fillna(0,inplace=True)
train_d_cumsum.fillna(0,inplace=True)


# In[38]:


train_d_mean.isnull().any()
# train_exp.isnull().any()
# train_log.isnull().any()
# train_d_cumsum.isnull().any()


# In[39]:


# train_diff_D=train_draw.set_index('FileID').groupby(level=0)['Day'].transform(lambda x : np.diff(x)).reset_index()


# In[40]:


train_diff_D_mean=train_draw.set_index('FileID').groupby(level=0)['Day'].transform(lambda x : np.sum(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()

train_diff_D_std=train_draw.set_index('FileID').groupby(level=0)['Day'].transform(lambda x : np.mean(np.diff(x)) if len(x)>1 
                                                                                   else np.diff(x)==0).groupby(level=0).first()


# In[41]:


train_diff_D_mean.head()


# In[42]:


train_diff_D_max=train_draw.set_index('FileID').groupby(level=0)['Day'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_D_min=train_draw.set_index('FileID').groupby(level=0)['Day'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[43]:


# train_diff_D_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[44]:



train_avg_D1=train_diff_D_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_D2=train_diff_D_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_D4=train_diff_D_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_D5=train_diff_D_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[45]:


del train_draw,train_diff_D_mean,train_diff_D_std,train_diff_D_max,train_diff_D_min


# In[ ]:


# full_merge=full_merge.sort_values('QueryTs')


# In[122]:


train_draw_dy=full_merge.groupby(["FileID","Day"])['QueryTs'].count().reset_index()


# In[123]:


train_draw_dy.index=train_draw_dy['FileID'].values


# In[124]:


train_draw_d=pd.DataFrame(train_draw_dy.groupby(["FileID"])["Day"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())


# In[4]:


train_draw_d.head()


# In[126]:


XC=[train_draw_d['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_d)) ]
train_draw_d['DD_99']=XC[:len(train_draw_d)]
train_draw_d['DD_01']=XC[len(train_draw_d):]


# In[127]:


train_draw_d=train_draw_d.drop(['<lambda>'],axis=1)


# In[128]:


train_draw_d.head()


# In[129]:


del XC


# In[130]:


train_diff_D_Dstd=train_draw_dy.groupby('FileID')['Day'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_D_Dmean=train_draw_dy.groupby('FileID')['Day'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[131]:


# train_diff_D_max.head()


# In[132]:


train_diff_D_Dmax=train_draw_dy.groupby('FileID')['Day'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_D_Dmin=train_draw_dy.groupby('FileID')['Day'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[133]:


del train_draw_dy


# In[134]:


train_diff_D_Dstd.shape


# # Hour
# 

# In[46]:


train_draw1=pd.DataFrame(full_merge.groupby(["FileID","Hour"])["Detected"].count()).reset_index()


# In[47]:


train_h_mean=train_draw1.set_index('FileID').groupby(level=0)['Detected'].agg({np.mean,np.min,np.max,np.std}).reset_index()
train_h_cumsum=train_draw1.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
# train_h_exp=train_draw1.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
# train_h_log=train_draw1.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
train_h_mean.fillna(0,inplace=True)
# train_h_exp.fillna(0,inplace=True)
# train_h_log.fillna(0,inplace=True)
train_h_cumsum.fillna(0,inplace=True)


# In[48]:


train_h_cumsum.isnull().any()


# In[49]:


train_diff_h_mean=train_draw1.set_index('FileID').groupby(level=0)['Hour'].transform(lambda x : np.sum(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_h_std=train_draw1.set_index('FileID').groupby(level=0)['Hour'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                   else np.diff(x)==0).groupby(level=0).first()


# In[50]:


train_diff_h_max=train_draw1.set_index('FileID').groupby(level=0)['Hour'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_h_min=train_draw1.set_index('FileID').groupby(level=0)['Hour'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[51]:


train_avg_h1=train_diff_h_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_h2=train_diff_h_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_h4=train_diff_h_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_h5=train_diff_h_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[52]:


del train_draw1, train_diff_h_mean,train_diff_h_std,train_diff_h_max,train_diff_h_min


# In[142]:


train_draw_hr=full_merge.groupby(["FileID","Hour"])['QueryTs'].count().reset_index()


# In[143]:



train_draw_h=pd.DataFrame(train_draw_hr.groupby(["FileID"])["Hour"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())


# In[144]:


XA=[train_draw_h['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_h)) ]
train_draw_h['DD_99']=XA[:len(train_draw_h)]
train_draw_h['DD_01']=XA[len(train_draw_h):]


# In[145]:


train_draw_h=train_draw_h.drop(['<lambda>'],axis=1)


# ##### 需要設置INDEX，才會有FILEID

# In[146]:


train_draw_hr.index=train_draw_hr['FileID'].values


# In[147]:


# train_draw_hr.head()


# In[148]:


train_diff_h_hstd=train_draw_hr.groupby(level=0)['Hour'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_h_hmean=train_draw_hr.groupby(level=0)['Hour'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_h_hmax=train_draw_hr.groupby(level=0)['Hour'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_h_hmin=train_draw_hr.groupby(level=0)['Hour'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[89]:


Train=pd.read_csv('Trend_temp.csv',index_col =0)


# In[92]:


Train['FileID'].unique


# # Minute

# In[53]:


train_draw2=pd.DataFrame(full_merge.groupby(["FileID","minute"])["Detected"].count()).reset_index()


# In[54]:


train_m_mean=train_draw2.set_index('FileID').groupby(level=0)['Detected'].agg({np.mean,np.min,np.max,np.std}).reset_index()
train_m_cumsum=train_draw2.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
# train_m_exp=train_draw2.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
# train_m_log=train_draw2.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
train_m_mean.fillna(0,inplace=True)
# train_m_exp.fillna(0,inplace=True)
# train_m_log.fillna(0,inplace=True)
train_m_cumsum.fillna(0,inplace=True)


# In[55]:


train_m_cumsum.isnull().any()


# In[56]:


train_diff_m_mean=train_draw2.set_index('FileID').groupby(level=0)['minute'].transform(lambda x : np.sum(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_m_std=train_draw2.set_index('FileID').groupby(level=0)['minute'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                   else np.diff(x)==0).groupby(level=0).first()


# In[57]:


train_diff_m_max=train_draw2.set_index('FileID').groupby(level=0)['minute'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_m_min=train_draw2.set_index('FileID').groupby(level=0)['minute'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[58]:


train_avg_m1=train_diff_m_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_m2=train_diff_m_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_m4=train_diff_m_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_m5=train_diff_m_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[59]:


del train_diff_m_mean,train_diff_m_std,train_draw2,train_diff_m_max,train_diff_m_min


# In[60]:


train_draw_mn=full_merge.groupby(["FileID","minute"])['QueryTs'].count().reset_index()
train_draw_m=pd.DataFrame(train_draw_mn.groupby(["FileID"])["minute"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
XM=[train_draw_m['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_m)) ]
train_draw_m['MM_99']=XM[:len(train_draw_m)]
train_draw_m['MM_01']=XM[len(train_draw_m):]
train_draw_m=train_draw_m.drop(['<lambda>'],axis=1)
train_draw_mn.index=train_draw_mn['FileID'].values
train_diff_m_mstd=train_draw_mn.groupby('FileID')['minute'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_m_mmean=train_draw_mn.groupby('FileID')['minute'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_m_mmax=train_draw_mn.groupby('FileID')['minute'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_m_mmin=train_draw_mn.groupby('FileID')['minute'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[158]:


# train_diff_m_mstd


# # Second

# In[ ]:


train_draw3=pd.DataFrame(full_merge.groupby(["FileID","QueryTs"])["Detected"].count()).reset_index()


# In[ ]:


train_s_mean=train_draw3.set_index('FileID').groupby(level=0)['Detected'].agg({np.mean,np.min,np.max,np.std}).reset_index()
train_s_cumsum=train_draw3.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
# train_s_exp=train_draw3.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
# train_s_log=train_draw3.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
train_s_mean.fillna(0,inplace=True)
# train_s_exp.fillna(0,inplace=True)
# train_s_log.fillna(0,inplace=True)
train_s_cumsum.fillna(0,inplace=True)


# In[ ]:


train_s_cumsum.isnull().any()


# In[ ]:



train_diff_qTs_mean=train_draw3.set_index('FileID').groupby(level=0)['QueryTs'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1
                                                                                          else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_qTs_std=train_draw3.set_index('FileID').groupby(level=0)['QueryTs'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                        else np.diff(x)==0).groupby(level=0).first()



# In[ ]:


train_diff_qTs_max=train_draw3.set_index('FileID').groupby(level=0)['QueryTs'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_qTs_min=train_draw3.set_index('FileID').groupby(level=0)['QueryTs'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[ ]:


train_avg_S1=train_diff_qTs_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_S2=train_diff_qTs_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_S4=train_diff_qTs_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_S5=train_diff_qTs_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[ ]:


del train_draw3,train_diff_qTs_mean,train_diff_qTs_std,train_diff_qTs_min,train_diff_qTs_max


# In[1]:


train_avg_S1.shape


# In[167]:


train_draw_sc=full_merge.groupby(["FileID"])['QueryTs'].count().reset_index()
train_draw_s=pd.DataFrame(train_draw_sc.groupby(["FileID"])["QueryTs"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
XM=[train_draw_s['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_s)) ]
train_draw_s['SS_99']=XM[:len(train_draw_s)]
train_draw_s['SS_01']=XM[len(train_draw_s):]
train_draw_s=train_draw_s.drop(['<lambda>'],axis=1)
train_draw_sc.index=train_draw_sc['FileID'].values
train_diff_s_sstd=train_draw_sc.groupby('FileID')['QueryTs'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_s_smean=train_draw_sc.groupby('FileID')['QueryTs'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_s_smax=train_draw_sc.groupby('FileID')['QueryTs'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_s_smin=train_draw_sc.groupby('FileID')['QueryTs'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[168]:


del XM,train_draw_sc


# # 10 Second

# In[169]:


train_draw4=pd.DataFrame(full_merge.groupby(["FileID","ten_second"])["Detected"].count()).reset_index()


# In[170]:


train_10s_mean=train_draw4.set_index('FileID').groupby(level=0)['Detected'].agg({np.mean,np.min,np.max,np.std}).reset_index()
train_10s_cumsum=train_draw4.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
# train_10s_exp=train_draw4.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
# train_10s_log=train_draw4.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
train_10s_mean.fillna(0,inplace=True)
# train_10s_exp.fillna(0,inplace=True)
# train_10s_log.fillna(0,inplace=True)
train_10s_cumsum.fillna(0,inplace=True)


# In[171]:


train_diff_10s_mean=train_draw4.set_index('FileID').groupby(level=0)['ten_second'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1
                                                                                          else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_10s_std=train_draw4.set_index('FileID').groupby(level=0)['ten_second'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                            else np.diff(x)==0).groupby(level=0).first()


# In[172]:


train_diff_10s_max=train_draw4.set_index('FileID').groupby(level=0)['ten_second'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_10s_min=train_draw4.set_index('FileID').groupby(level=0)['ten_second'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()



# In[173]:


train_avg_10S1=train_diff_10s_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_10S2=train_diff_10s_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_10S4=train_diff_10s_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_10S5=train_diff_10s_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[174]:


del train_draw4,train_diff_10s_mean,train_diff_10s_std,train_diff_10s_max,train_diff_10s_min


# In[175]:


full_merge.columns


# In[244]:


train_draw_10sc=full_merge.groupby(["FileID","ten_second"])['QueryTs'].count().reset_index()
train_draw_10s=pd.DataFrame(train_draw_10sc.groupby(["FileID"])["ten_second"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
XM=[train_draw_10s['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_10s)) ]
train_draw_10s['10S_99']=XM[:len(train_draw_10s)]
train_draw_10s['10S_01']=XM[len(train_draw_10s):]
train_draw_10s=train_draw_10s.drop(['<lambda>'],axis=1)
train_draw_10sc.index=train_draw_10sc['FileID'].values
train_diff_10s_sstd=train_draw_10sc.groupby('FileID')['ten_second'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_10s_smean=train_draw_10sc.groupby('FileID')['ten_second'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_10s_smax=train_draw_10sc.groupby('FileID')['ten_second'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_10s_smin=train_draw_10sc.groupby('FileID')['ten_second'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[245]:


del XM,train_draw_10sc


# # Customer ID

# In[37]:


train_draw5=pd.DataFrame(full_merge.groupby(["FileID","CustomerID",])["Detected"].count()).reset_index()


# In[179]:


train_customer_mean = train_draw5.set_index('FileID').groupby(level=0).agg([np.mean,np.min,np.max,np.std]).reset_index()
train_customer_cumsum=train_draw5.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
# train_customer_exp=train_draw5.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
# train_customer_log=train_draw5.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
train_customer_mean.fillna(0,inplace=True)
# train_customer_exp.fillna(0,inplace=True)
# train_customer_log.fillna(0,inplace=True)
train_customer_cumsum.fillna(0,inplace=True)


# In[180]:


train_diff_customer_mean=train_draw5.set_index('FileID').groupby(level=0).transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1
                                                                                          else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_customer_std=train_draw5.set_index('FileID').groupby(level=0).transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                            else np.diff(x)==0).groupby(level=0).first()


# In[181]:


train_diff_customer_max=train_draw5.set_index('FileID').groupby(level=0).transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_customer_min=train_draw5.set_index('FileID').groupby(level=0).transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[182]:


train_avg_cus1=train_diff_customer_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_cus2=train_diff_customer_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_cus4=train_diff_customer_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_cus5=train_diff_customer_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[183]:


del train_draw5,train_diff_customer_mean,train_diff_customer_std,train_diff_customer_min,train_diff_customer_max


# In[184]:


# train_draw5.head()


# In[185]:


train_avg_cus1.head()


# In[187]:


# 


# In[257]:


train_draw_customer=full_merge.groupby(["FileID","CustomerID"])['QueryTs'].count().reset_index()
train_draw_customer.index=train_draw_customer['FileID'].values
train_draw_cus=pd.DataFrame(train_draw_customer.groupby(level=0).agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
train_draw_cus.columns= pd.Index([e[0] + e[1] for e in train_draw_cus.columns.tolist()])
XF=[train_draw_cus['QueryTs<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_cus)) ]
train_draw_cus['CU_99']=XF[:len(train_draw_cus)]
train_draw_cus['CU_01']=XF[len(train_draw_cus):]
train_draw_cus=train_draw_cus.drop(['QueryTs<lambda>'],axis=1)
train_draw_cus=train_draw_cus.rename(columns={"index": "FileID"})
train_diff_cus_sstd=train_draw_customer.groupby(level=0).transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_cus_smean=train_draw_customer.groupby(level=0).transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_cus_smax=train_draw_customer.groupby(level=0).transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})
train_diff_cus_smin=train_draw_customer.groupby(level=0).transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


# In[265]:


# train_draw_cus=train_draw_cus.rename(columns={"index": "FileID"})


# In[266]:


train_draw_cus.columns


# In[189]:


del XF,train_draw_customer


# # 30 Second

# In[190]:


train_draw6=pd.DataFrame(full_merge.groupby(["FileID","thirty_second"])["Detected"].count()).reset_index()


# In[191]:


train_30s_mean = train_draw6.set_index('FileID').groupby(level=0).agg([np.mean,np.min,np.max,np.std]).reset_index()
train_30s_cumsum = train_draw6.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
#train_draw6.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
#train_draw6.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
#train_customer_mean.fillna(0,inplace=True)
# train_customer_exp.fillna(0,inplace=True)
# train_customer_log.fillna(0,inplace=True)
train_30s_mean.fillna(0,inplace=True)
train_30s_cumsum.fillna(0,inplace=True)


# In[192]:


train_diff_30s_mean=train_draw6.set_index('FileID').groupby(level=0)['thirty_second'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1
                                                                                          else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_30s_std=train_draw6.set_index('FileID').groupby(level=0)['thirty_second'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                            else np.diff(x)==0).groupby(level=0).first()


# In[193]:


train_diff_30s_max=train_draw6.set_index('FileID').groupby(level=0)['thirty_second'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_30s_min=train_draw6.set_index('FileID').groupby(level=0)['thirty_second'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[194]:


train_avg_30S1=train_diff_30s_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_30S2=train_diff_30s_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_30S4=train_diff_30s_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_30S5=train_diff_30s_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[195]:


del train_draw6,train_diff_30s_mean,train_diff_30s_std,train_diff_30s_max,train_diff_30s_min


# In[196]:


train_30s_cumsum .columns


# In[ ]:


train_avg_30S2.isnull().any()


# In[198]:


train_draw_30sc=full_merge.groupby(["FileID","thirty_second"])['QueryTs'].count().reset_index()
train_draw_30s=pd.DataFrame(train_draw_30sc.groupby(["FileID"])["thirty_second"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
XA=[train_draw_30s['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_30s)) ]
train_draw_30s['30S_99']=XA[:len(train_draw_30s)]
train_draw_30s['30S_01']=XA[len(train_draw_30s):]
train_draw_30s=train_draw_30s.drop(['<lambda>'],axis=1)
train_draw_30sc.index=train_draw_30sc['FileID'].values
train_diff_30s_sstd=train_draw_30sc.groupby('FileID')['thirty_second'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_30s_smean=train_draw_30sc.groupby('FileID')['thirty_second'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


train_diff_30s_smax=train_draw_30sc.groupby('FileID')['thirty_second'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_30s_smin=train_draw_30sc.groupby('FileID')['thirty_second'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})



# In[262]:


# train_draw_30s.head()


# In[199]:


del XA,train_draw_30sc


# # 6 Hour

# In[200]:


train_draw7=pd.DataFrame(full_merge.groupby(["FileID","six_hour"])["Detected"].count()).reset_index()


# In[201]:


train_6h_mean = train_draw7.set_index('FileID').groupby(level=0).agg([np.mean,np.min,np.max,np.std]).reset_index()
train_6h_cumsum = train_draw7.set_index('FileID').groupby(level=0)['Detected'].agg({np.cumsum}).reset_index().groupby('FileID').agg({np.mean,np.min,np.max,np.std}).reset_index()
#train_draw6.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.exp(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
#train_draw6.set_index('FileID').groupby(level=0)['Detected'].transform(lambda x : np.log1p(x)).reset_index().groupby(level=0).agg({np.mean,np.min,np.max}).reset_index()
#train_customer_mean.fillna(0,inplace=True)
# train_customer_exp.fillna(0,inplace=True)
# train_customer_log.fillna(0,inplace=True)
train_6h_mean.fillna(0,inplace=True)
train_6h_cumsum.fillna(0,inplace=True)


# In[202]:


train_diff_6h_mean=train_draw7.set_index('FileID').groupby(level=0)['six_hour'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1
                                                                                          else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_6h_std=train_draw7.set_index('FileID').groupby(level=0)['six_hour'].transform(lambda x : np.std(np.diff(x)) if len(x)>1 
                                                                                            else np.diff(x)==0).groupby(level=0).first()


# In[203]:


train_diff_6h_max=train_draw7.set_index('FileID').groupby(level=0)['six_hour'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()
train_diff_6h_min=train_draw7.set_index('FileID').groupby(level=0)['six_hour'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first()


# In[204]:


train_avg_6H1=train_diff_6h_mean.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_6H2=train_diff_6h_std.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_6H4=train_diff_6h_max.reset_index().groupby(["FileID"]).mean().reset_index()
train_avg_6H5=train_diff_6h_min.reset_index().groupby(["FileID"]).mean().reset_index()


# In[205]:


del train_draw7,train_diff_6h_mean,train_diff_6h_std,train_diff_6h_max,train_diff_6h_min


# In[206]:


train_6h_mean.head()


# In[207]:


train_draw_6hr=full_merge.groupby(["FileID","six_hour"])['QueryTs'].count().reset_index()
train_draw_6h=pd.DataFrame(train_draw_6hr.groupby(["FileID"])["six_hour"].agg({'mean','min','max','std',lambda x : [np.percentile(x,99),np.percentile(x,1)]}).reset_index())
XA=[train_draw_6h['<lambda>'][i][j] for j in range(2) for i in range(len(train_draw_6h)) ]
train_draw_6h['6H_99']=XA[:len(train_draw_6h)]
train_draw_6h['6H_01']=XA[len(train_draw_6h):]
train_draw_6h=train_draw_6h.drop(['<lambda>'],axis=1)
train_draw_6hr.index=train_draw_6hr['FileID'].values
train_diff_6h_hstd=train_draw_6hr.groupby('FileID')['six_hour'].transform(lambda x : np.std(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_6h_hmean=train_draw_6hr.groupby('FileID')['six_hour'].transform(lambda x : np.mean(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})


train_diff_6h_hmax=train_draw_6hr.groupby('FileID')['six_hour'].transform(lambda x : np.max(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})

train_diff_6h_hmin=train_draw_6hr.groupby('FileID')['six_hour'].transform(lambda x : np.min(np.diff(x))/(len(x)-1) if len(x)>1 
                                                                                   else np.diff(x)/(len(x))).groupby(level=0).first().reset_index().rename(columns={"index": "FileID"})



# In[210]:


del XA,train_draw_6hr


# In[211]:


train_diff_6h_hmin.head()


# In[ ]:


# train_draw['Detected'].sum()


# In[ ]:


# testt=train_draw5.values


# In[ ]:


# try:
#     train_draw=train_draw.drop(['level_0'],axis=1)
#     train_draw=train_draw.drop(['index'],axis=1)
# except:
#     print ("ok")
    


# In[ ]:



# train_avg_m1、train_avg_S1、train_avg_10S1、train_avg_cus1、train_avg_30S1、train_avg_6H1


# ## ProductID

# In[38]:


train_draw_pd=full_merge.groupby(["FileID","ProductID"])['QueryTs'].count().reset_index()


# In[39]:


train_draw_pd['QueryTs']=train_draw_pd['QueryTs'].clip(0,1)


# In[40]:


train_draw_pp=train_draw_pd.groupby(["FileID"])['QueryTs'].count().reset_index()


# In[41]:


train_draw_pp.head()


# # Section 2

# # Day Feature

# In[42]:


try :
    train_customer=train_customer.drop(["level_0","index"],axis=1)
except:
    print ("ok")


# In[43]:


train_customer_mean.head()


# In[44]:


train_customer_mean.columns


# In[45]:


train_customer_cumsum.columns= pd.Index([e[0] + e[1] for e in train_customer_cumsum.columns.tolist()])
train_customer_mean.columns = pd.Index([e[0] + e[1] for e in train_customer_mean.columns.tolist()])
train_10s_cumsum.columns= pd.Index([e[0] + e[1] for e in train_10s_cumsum.columns.tolist()])
train_30s_cumsum.columns= pd.Index([e[0] + e[1] for e in train_30s_cumsum.columns.tolist()])
train_30s_mean.columns= pd.Index([e[0] + e[1] for e in train_30s_mean.columns.tolist()])
train_6h_mean.columns= pd.Index([e[0] + e[1] for e in train_6h_mean.columns.tolist()])
train_6h_cumsum.columns= pd.Index([e[0] + e[1] for e in train_6h_cumsum.columns.tolist()])


# In[235]:


train_d_cumsum.columns= pd.Index([e[0] + e[1] for e in train_d_cumsum.columns.tolist()])
train_h_cumsum.columns= pd.Index([e[0] + e[1] for e in train_h_cumsum.columns.tolist()])
train_m_cumsum.columns= pd.Index([e[0] + e[1] for e in train_m_cumsum.columns.tolist()])
train_s_cumsum.columns= pd.Index([e[0] + e[1] for e in train_s_cumsum.columns.tolist()])


# In[1]:


train_avg_D5.columns


# In[267]:


f_list =[train_d_mean,train_d_cumsum,
         train_h_mean,train_h_cumsum,
         train_m_mean,train_m_cumsum,
         train_s_mean,train_s_cumsum,
         train_10s_mean,train_10s_cumsum,
         train_30s_mean,train_30s_cumsum,
         train_6h_mean,train_6h_cumsum,
         train_customer_mean,train_customer_cumsum,
         train_avg_D1, train_avg_D2, train_avg_D4,train_avg_D5,
         train_avg_h1, train_avg_h2 , train_avg_h4,train_avg_h5,
         train_avg_m1, train_avg_m2, train_avg_m4,train_avg_m5,
         train_avg_S1, train_avg_S2, train_avg_S4,train_avg_S5,
         train_avg_10S1 , train_avg_10S2, train_avg_10S4,train_avg_10S5,
         train_avg_30S1,train_avg_30S2,train_avg_30S4,train_avg_30S5,
         train_avg_6H1,train_avg_6H2, train_avg_6H4,train_avg_6H5,
         train_avg_cus1,train_avg_cus2, train_avg_cus4,train_avg_cus5,
         train_draw_d, train_diff_D_Dstd, train_diff_D_Dmean, train_diff_D_Dmax, train_diff_D_Dmin,
         train_draw_h, train_diff_h_hstd, train_diff_h_hmean, train_diff_h_hmax, train_diff_h_hmin,
         train_draw_m, train_diff_m_mstd, train_diff_m_mmean, train_diff_m_mmax, train_diff_m_mmin,
         train_draw_s, train_diff_s_sstd, train_diff_s_smean, train_diff_s_smax, train_diff_s_smin,
         train_draw_10s, train_diff_10s_sstd, train_diff_10s_smean, train_diff_10s_smax, train_diff_10s_smin,
         train_draw_cus, train_diff_cus_sstd, train_diff_cus_smean, train_diff_cus_smax, train_diff_cus_smin,
         train_draw_30s, train_diff_30s_sstd, train_diff_30s_smean, train_diff_30s_smax, train_diff_30s_smin,
         train_draw_6h, train_diff_6h_hstd, train_diff_6h_hmean, train_diff_6h_hmax, train_diff_6h_hmin,
         train_draw_pp]
temp =pd.DataFrame(train_h_mean['FileID'])
sum=0
for i in f_list[::-1]:
    print (sum,i.columns)
    temp = pd.merge(i, temp, on='FileID' , how ='inner')
    sum+=1


# In[85]:


T=pd.read_csv('training_0301_2_y.csv')


# In[86]:





# In[269]:


temp.to_csv("Trend_temp.csv")


# In[240]:


# #f_list =[train_d_mean,train_d_min,train_d_max,train_d_std,
#          train_h_mean,train_d_min,train_d_max,train_d_std,
#          train_m_mean,train_m_min,train_m_max,train_m_std,
#          train_s_mean,train_s_min,train_s_max,train_s_std,
#          train_10s_mean,train_10s_min,train_10s_max,train_10s_std,
#          train_10s_mean,train_10s_min,train_10s_max,train_10s_std,
#          train_6h_mean,train_6h_min,train_6h_max,train_6h_std,
#          train_cus_mean,train_cus_min,train_cus_max,train_cus_std,
#          train_avg_D1, train_avg_D2, train_avg_h1, train_avg_h2 , train_avg_m1, 
#          train_avg_m2, train_avg_S1, train_avg_S2, train_avg_10S1 , train_avg_10S2,
#          train_avg_30S1,train_avg_30S2,train_avg_6H1,train_avg_6H2,train_avg_cus1,train_avg_cus2 ]


# In[241]:


# f_list =[train_d_mean,train_d_cumsum,
#          train_h_mean,train_h_cumsum,
#          train_m_mean,train_m_cumsum,
#          train_s_mean,train_s_cumsum,
#          train_10s_mean,train_10s_cumsum,
#          train_30s_mean,train_30s_cumsum,
#          train_6h_mean,train_6h_cumsum,
#          train_customer_mean,train_customer_cumsum,
#          train_avg_D1, train_avg_D2, train_avg_h1, train_avg_h2 , train_avg_m1, 
#          train_avg_m2, train_avg_S1, train_avg_S2, train_avg_10S1 , train_avg_10S2,
#          train_avg_30S1,train_avg_30S2,train_avg_6H1,train_avg_6H2,train_avg_cus1,train_avg_cus2 ]


# In[270]:


temp.columns[:5]


# In[271]:


temp.columns=["FileID","d_std", "d_mean","d_max","d_min","d_cumsum_std","d_cumsum_mean","d_cumsum_max","d_cumsum_min",
              "h_std", "h_mean","h_max","h_min","h_cumsum_std","h_cumsum_mean","h_cumsum_max","h_cumsum_min",
              "m_std","m_mean","m_max","m_min","m_cumsum_std","m_cumsum_mean","m_cumsum_max","m_cumsum_min",
              "s_std","s_mean","s_max","s_min","s_cumsum_std","s_cumsum_mean","s_cumsum_max","s_cumsum_min",
              "10s_std","10s_mean","10s_max","10s_min","10s_cumsum_std","10s_cumsum_mean","10s_cumsum_max","10s_cumsum_min",
              "30s_mean2","30s_min2","30s_max2","30s_std2","30s_mean","30s_min","30s_max","30s_std",
              "30s_cumsum_std","30s_cumsum_mean","30s_cumsum_max","30s_cumsum_min",
              "6h_mean2","6h_min2","6h_max2","6h_std2","6h_mean","6h_min","6h_max","6h_std",
              "6h_cumsum_std","6h_cumsum_mean","6h_cumsum_max","6h_cumsum_min",            
              "customer_mean","customer_min","customer_max","customer_std",
              "customer_std_cumsum","customer_mean_cumsum","customer_max_cumsum","customer_min_cumsum",
              "Day_mean","Day_std","Day_max","Day_min",
              "Hour_mean","Hour_std","Hour_max","Hour_min",
              "Min_mean","Min_std","Min_max","Min_min",
              "QueryTs_mean","QueryTs_std","QueryTs_max","QueryTs_min",
              "10_QueryTs_mean","10_QueryTs_std","10_QueryTs_max","10_QueryTs_min",
              "30_QueryTs_mean","30_QueryTs_std","30_QueryTs_max","30_QueryTs_min",
              "6_Hour_mean","6_Hour_std","6_Hour_max","6_Hour_min",
              "Cus_d_mean","Cus_d_std","Cus_d_max", "Cus_d_min",
              "train_draw_d1","train_draw_d2","train_draw_d3","train_draw_d4","train_draw_d5","train_draw_d6",
              "train_diff_D_Dstd","train_diff_D_Dmean","train_diff_D_Dmax","train_diff_D_Dmin",
              "train_draw_h1","train_draw_h2","train_draw_h3","train_draw_h4","train_draw_h5","train_draw_h6",
              "train_diff_h_hstd","train_diff_h_hmean","train_diff_h_hmax","train_diff_h_hmin",
              "train_draw_m1","train_draw_m2","train_draw_m3","train_draw_m4","train_draw_m5","train_draw_m6",
              "train_diff_m_mstd","train_diff_m_mean","train_diff_m_max","train_diff_m_min",
              "train_draw_s1","train_draw_s2","train_draw_s3","train_draw_s4","train_draw_s5","train_draw_s6",
              "train_diff_s_sstd","train_diff_s_smean","train_diff_s_smax","train_diff_s_smin",
              "train_draw_10s1","train_draw_10s2","train_draw_10s3","train_draw_10s4","train_draw_10s5","train_draw_10s6",
              "train_diff_10s_sstd","train_diff_10s_smean","train_diff_10s_smax","train_diff_10s_smin",
              "train_draw_cus1","train_draw_cus2","train_draw_cus3","train_draw_cus4","train_draw_cus5","train_draw_cus6",
              "train_diff_cus_sstd","train_diff_cus_smean","train_diff_cus_smax","train_diff_cus_smin",
              "train_draw_30s1","train_draw_30s2","train_draw_30s3","train_draw_30s4","train_draw_30s5","train_draw_30s6",
              "train_diff_30s_sstd","train_diff_30s_smean","train_diff_30s_smax","train_diff_30s_smin",
              "train_draw_6h1","train_draw_6h2","train_draw_6h3","train_draw_6h4","train_draw_6h5","train_draw_6h6",
              "train_diff_6h_hstd","train_diff_6h_hmean","train_diff_6h_hmax","train_diff_6h_hmin","train_draw_pp" ]
#"train_customer4","train_customer5","train_customer6",


# In[272]:


temp.fillna(0,inplace=True)


# In[273]:


temp.to_csv("Trend_temp.csv")


# # Product ID Feature （full merge完了以後直接接這個）
# # run from here 3/19

# In[325]:


temp=pd.read_csv('Trend_temp_0317.csv')


# In[326]:


temp=temp.drop(['Unnamed: 0'],axis=1)


# In[327]:


temp.head()


# In[328]:


full_merge['ProductID']=full_merge['ProductID'].astype(str)


# In[329]:


temp=temp.loc[:,["FileID",
              "train_draw_d1","train_draw_d2","train_draw_d3","train_draw_d4","train_draw_d5","train_draw_d6",
              "train_diff_D_Dstd","train_diff_D_Dmean","train_diff_D_Dmax","train_diff_D_Dmin",
              "train_draw_h1","train_draw_h2","train_draw_h3","train_draw_h4","train_draw_h5","train_draw_h6",
              "train_diff_h_hstd","train_diff_h_hmean","train_diff_h_hmax","train_diff_h_hmin",
              "train_draw_m1","train_draw_m2","train_draw_m3","train_draw_m4","train_draw_m5","train_draw_m6",
              "train_diff_m_mstd","train_diff_m_mean","train_diff_m_max","train_diff_m_min",
              "train_draw_s1","train_draw_s2","train_draw_s3","train_draw_s4","train_draw_s5","train_draw_s6",
              "train_diff_s_sstd","train_diff_s_smean","train_diff_s_smax","train_diff_s_smin",
              "train_draw_10s1","train_draw_10s2","train_draw_10s3","train_draw_10s4","train_draw_10s5","train_draw_10s6",
              "train_diff_10s_sstd","train_diff_10s_smean","train_diff_10s_smax","train_diff_10s_smin",
              "train_draw_cus1","train_draw_cus2","train_draw_cus3","train_draw_cus4","train_draw_cus5","train_draw_cus6",
              "train_diff_cus_sstd","train_diff_cus_smean","train_diff_cus_smax","train_diff_cus_smin",
              "train_draw_30s1","train_draw_30s2","train_draw_30s3","train_draw_30s4","train_draw_30s5","train_draw_30s6",
              "train_diff_30s_sstd","train_diff_30s_smean","train_diff_30s_smax","train_diff_30s_smin",
              "train_draw_6h1","train_draw_6h2","train_draw_6h3","train_draw_6h4","train_draw_6h5","train_draw_6h6",
              "train_diff_6h_hstd","train_diff_6h_hmean","train_diff_6h_hmax","train_diff_6h_hmin","train_draw_pp"]]


# In[330]:


temp.head()


# In[331]:


# def single_max(ci):
#     return ci.value_counts().max() 
# def positive_mean(ci): 
#     cnt = ci.value_counts() 
#     return cnt[cnt>0].mean() 

# pivot_ci = full_merge.pivot_table(values='CustomerID', columns='ProductID', index=['FileID'], aggfunc=[pd.Series.count, pd.Series.nunique, single_max, positive_mean]) 

# pivot_ci.columns = pd.MultiIndex.from_tuples([('ci_'+x,y) for (x,y) in pivot_ci.columns.ravel()]) 
# pivot_ci.fillna(0, inplace=True) 
# pivot_ci.head(3) 
# print('CustomerID pivot done.')


# In[332]:


# pivot_ci.to_csv('pivot.csv')


# In[333]:


# pivot_ci.columns=pd.Index([e[0] + e[1] for e in pivot_ci.columns.tolist()]) 


# In[334]:


# pivot_ci=pivot_ci.reset_index()


# In[335]:


pivot_0=pd.read_csv('pivot.csv',index_col=0)


# In[336]:


if 'Unnamed: 0' in pivot_0.columns:
    pivot_0=pivot_0.drop(['Unnamed: 0'],axis=1)


# In[337]:


pivot_0.head()


# In[47]:


R=pivot_0.sum(axis=0)


# In[93]:


R[R.index[1]]


# In[51]:


T=pd.DataFrame()
for u in range(1,len(R.index)):
    T[u]=(pivot_0.iloc[:,[u]]/R[R.index[u]])


# In[ ]:


T.shape
# pivot_0.sum(axis=1)


# In[338]:


feature = pd.pivot_table(full_merge[['FileID','ProductID']], index =['FileID'], columns= ['ProductID'], aggfunc=len,)


# In[339]:


feature.fillna(0,inplace=True)


# In[340]:


# F=feature.sum(axis=0)


# In[ ]:


# T=pd.DataFrame()
# for u in range(1,len(R.index)):
#     T[u]=(feature.iloc[:,[u]]/R[R.index[u]])



# In[ ]:


feature['FileID'] = feature.index


# In[50]:


# train_multi_merge=train_custome.columns.values


# In[51]:


# temp1=pd.merge(feature,train_customer,on='FileID' , how ='inner')


# In[52]:


# temp2=pd.merge(temp1,train_avg_cus1,on='FileID' , how ='inner')


# # Week、Day
# # 先不要跑王董的

# In[44]:



#training set time label
# full_merge["QueryTs_hr"] =full_merge["QueryTs"].map(lambda x: datetime.datetime.fromtimestamp(x).strftime('%H'))
# full_merge["QueryTs_week"] =full_merge["QueryTs"].map(lambda x: datetime.datetime.fromtimestamp(x).isoweekday())


#Create training set
# ProductID_arr = pd.crosstab(full_merge.FileID,full_merge.ProductID, margins=False)
Hour_arr = pd.crosstab(full_merge.FileID,full_merge.Hour, margins=False)
Week_arr = pd.crosstab(full_merge.FileID,full_merge.Week, margins=False)

#CustomID



# In[ ]:


Hour_arr=Hour_arr.reset_index()
Week_arr=Week_arr.reset_index()


# In[ ]:


Customer_num= full_merge.groupby('FileID')['CustomerID'].unique()


# In[ ]:


Customer_num.head()


# In[ ]:


Customer_arr=[len(Customer_num[i]) for i in range(len(Customer_num))]


# In[65]:


# Customer_arr


# In[66]:


data_index = list(temp.index)
Cus_num = pd.DataFrame(Customer_arr, columns=['Customer_num'], index=data_index)
Cus_num['FileID']=Week_arr['FileID']


# In[67]:


Week_arr.shape


# In[68]:


Wang=pd.concat([Week_arr,Cus_num],ignore_index=True,axis=1,join='inner',names=['wang_week_hour_cusID'+str(range(Week_arr.shape[1]+Cus_num.shape[1]))])   


# In[69]:


Wang=Wang.drop([16],axis=1)


# In[70]:


Wang.head()


# In[71]:


Wang.columns=['wang_week_hour_cusID'+str(i) for i in range(len(Wang.columns))]


# In[72]:


Wang=Wang.rename(columns={'wang_week_hour_cusID0':'FileID'})


# In[73]:


Wang.head()


# In[76]:


Wang.to_csv('second.csv')


# In[341]:


Wang=pd.read_csv('second.csv')


# In[342]:


temp.shape


# In[343]:


temp=pd.merge(temp,Wang,on=['FileID'])


# In[344]:


temp.head()


# In[173]:


f_list2=[Wang,pivot_0]


# In[175]:


sum=0
for i in f_list2[::-1]:
    print (sum,i.columns)
    temp = pd.merge(i, temp, on='FileID' , how ='inner')
    sum+=1


# In[75]:


temp.head()


# # RUN!!!!!!

# In[345]:


f_log = pd.merge(temp, feature, on = ['FileID'], how ='inner')


# In[349]:


f_log=temp.copy()


# In[350]:


f_log.head()


# In[351]:


temp.head()


# In[352]:


del full_merge


# # Everything with productID Feature
# # 可跑一下

# In[73]:


col_important=["train_draw_d1","train_draw_d2","train_draw_d3","train_draw_d4","train_draw_d5","train_draw_d6",
              "train_diff_D_Dstd","train_diff_D_Dmean","train_diff_D_Dmax","train_diff_D_Dmin",
              "train_diff_6h_hstd","train_diff_6h_hmean"]
col_names=temp.columns
# col_names=col_names.drop(["FileID"])
col_productID=feature.columns
col_productID=col_productID.drop(["FileID"])


# In[74]:


len(temp.columns)


# In[75]:


# custom_name=["customer_mean","customer_min","customer_max","customer_std",  "Cus_d_mean","Cus_d_std","Cus_d_max", "Cus_d_min"]


# In[76]:


# for i in custom_name:
#     for j in col_names[1:]:
#         name=i+j
#         f_log[name]=f_log[i]* f_log[j]


# In[77]:


# custom_name


# In[78]:


count=0
for i in col_names[1:]:
    for j in col_productID:
#         print (i,j)
        name=str(i)+str(j)
        f_log[name]=f_log[i]*f_log[j]
        count+=1
print (len(f_log.columns))


# In[79]:


print (i,j)


# In[80]:


# f_log['10s_cumsum_std'].head()


# In[81]:


f_log.head()


# In[82]:


def safe_ln(x, minval=0.0000000001):
    return np.log1p(x.clip(minval,x))


# In[83]:


for i in col_important:
    for j in col_productID:
        name="power_2"+str(i)+j
        f_log[name]=(f_log[j])*(f_log[i]**2)
        name2="log"+str(i)+j
        f_log[name2]=(f_log[j])*safe_ln(f_log[i])


# In[ ]:


f_log.fillna(999999,inplace=True)


# In[ ]:


f_log.columns


# In[353]:


Lang=pickle.load( open ( "pivot_3.pickle" , "rb" ))


# In[354]:


# df_num=df_num[df_num['bool']==True]


# In[355]:


f_log.isnull().any().any()


# In[ ]:


# f_log[f_log.loc[:,'logtrain_draw_d10374c4'].isnull()]


# In[ ]:


# f_log['logtrain_diff_6h_hmindd8d4a'].isnull()


# # Log All
# # LOG 和以下的1/sqrt擇一

# In[92]:


f_ll=pd.DataFrame(np.log1p(f_log.iloc[:,np.array(list(range(1,len(f_log.columns))))]))


# In[93]:


f_ll.columns=list("loggg"+name for name in f_log.columns[1:])


# In[94]:


# f_ll['FileID']=f_log['FileID']


# In[95]:


f_ll.fillna(0,inplace=True)


# In[96]:


# f_log=pd.concat([f_log,f_ll],axis=1)


# In[97]:


f_ll.isnull().any().any()


# In[98]:


# f_log.head()


# In[99]:


# f_log['FileID']


# In[100]:


f_log=pd.concat([f_log,f_ll],axis=1)


# In[101]:


f_log.head()


# # Sqrt all

# In[49]:


# np.sqrt(f_log.iloc[:,1])


# In[50]:


# (list(range(1,3213)))


# In[116]:


f_oversqrt=pd.DataFrame(1/np.sqrt(f_log.iloc[:,np.array(list(range(1,len(f_log.columns))))]))


# In[117]:


f_oversqrt.columns=list("sqrt_"+name for name in f_log.columns[1:len(f_log.columns)])


# In[118]:


f_oversqrt.columns


# In[119]:


f_log=f_oversqrt.copy()


# In[6]:


# f_log=pd.concat([f_log,f_oversqrt],axis=1)


# In[5]:


f_log=f_log.replace(np.inf, 9999)


# In[ ]:


f_log.shape


# In[ ]:


# full_merge


# In[57]:


Tu=pd.read_csv('Trend_temp.csv')


# In[58]:


f_log['FileID']=Tu['FileID']


# In[59]:


f_log.head()


# In[60]:


# f_log2=pd.DataFrame()
# for i in f_log.columns[756+753:]:
#     if i in f_log.columns:
#         f_log2[i]=f_log[i]


# In[61]:


# f_log2.head()


# In[62]:


# f_log=f_log2.copy()


# In[63]:


# f_log.head()


# # Merge f_log and train,test （跑）

# In[356]:


f_log.head()


# In[357]:


import pickle
loaded_model = pickle.load(open("pivot_3.pickle", "rb"))


# In[358]:


loaded_model=loaded_model.reset_index()


# In[359]:


loaded_model.columns=pd.Index([e[0]+e[1] for e in loaded_model.columns.tolist()])


# In[360]:


f_log.head()


# In[361]:


f_log=pd.merge(loaded_model,f_log,on=['FileID'])


# In[362]:


f_log.head()


# In[363]:


test.shape


# In[364]:


train_data = pd.merge(f_log, train, on = ['FileID'], how ='inner')
test_data = pd.merge(f_log, test, on = ['FileID'], how ='inner')


# In[365]:


train_data.drop('Detected', axis = 1, inplace= True)
test_data.drop('Detected', axis = 1, inplace= True)


# In[366]:


# train_data.to_csv("training_0303_4_x.csv",index=False)
# test_data.to_csv("testing_0303_4_x.csv",index=False)


# In[367]:


count = 0
for i in range(train_data.shape[0]):
    if train_data['FileID'][i] == train['FileID'][i]:
        count+=1
print(test_data.shape[0])
print(train_data.shape[0])


# In[368]:


f_log.head()


# In[369]:


# train_data.head()


# # Drop Nonimportant featrue
# # Don't run this!!!!!!!

# In[78]:


# import pickle
# loaded_model = pickle.load(open("xgboost_0948_pickle.dat", "rb"))
# feature_imp=pd.read_csv("0948_log.csv",names=["important"],skiprows=[0])


# In[79]:


# feature_imp.head()


# In[80]:


# sort_df=feature_imp.sort_values(by=["important"])


# In[81]:


# numpy.random.randint


# In[82]:


# zero_name=sort_df[sort_df["important"==0]].index


# In[83]:


# sort_df.drop(zero_name,axis=1)#[zeros_name]


# In[86]:


f_log.head()


# # Model

# In[370]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from matplotlib import pyplot as plt


# In[371]:


# train_data


# In[372]:


#Convert data frame to np array

# X = data_log[""].values()
# y = data_log["Detected"].values()
import pandas as pd

x_train=train_data.copy()
x_test=test_data.copy()
y_train=pd.read_csv("training_0301_2_y.csv")


# In[373]:


x_test.head()


# In[374]:


x_train=x_train.drop(["FileID"],axis=1)
y_train=y_train.drop(["FileID"],axis=1)


# In[375]:


x_test=x_test.drop(["FileID"],axis=1)


# In[377]:


x_test.head()


# # Scaler 可跑可不跑

# In[94]:


x_train.isnull().any().any()


# In[100]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train) 
x_train=pd.DataFrame(x_train)


# In[ ]:


scaler = preprocessing.StandardScaler().fit(x_test)
x_test=scaler.transform(x_test) 
x_test=pd.DataFrame(x_test)


# # RUN!!!!!!!!!!!!!!!!!

# In[378]:


x_test1=x_test


# In[379]:


x_test1.head()


# In[380]:


x_test.head()


# In[381]:



# x_train=x_train.dropna(x_train)
# x_train["Day_mean"]=x_train["Day_mean"].fillna(x_train["Day_mean"].mean())
# x_train["Day_std"]=x_train["Day_std"].fillna(x_train["Day_std"].mean())
# x_train["QueryTs_mean"]=x_train["QueryTs_mean"].fillna(x_train["Day_mean"].mean())
# x_train["QueryTs_std"]=x_train["QueryTs_std"].fillna(x_train["Day_mean"].mean())

# split our training data (X,y) into trainning & validation

# x_train.head()


# In[382]:


# X_train, X_vad, Y_train, Y_vad = train_test_split(x_train, y_train, random_state = 18, test_size = 0.25)


# In[383]:


from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_validation import KFold

from sklearn.metrics import roc_curve

from multiprocessing import Process, Pool
import functools

import re
import unidecode
import math

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb


# In[410]:


fold = 2 # 手動設置要幾個fold
        
        # ==== 以下建議搭配slide 圖示會更清楚運作過程 ====
        # ==== 以下建議搭配slide 圖示會更清楚運作過程 ====
        # ==== 以下建議搭配slide 圖示會更清楚運作過程 ====

def oof(model, ntrain, ntest, kf, train, labels, test):
    # model, 用的模型
    # ntrain, 訓練集的row number
    # ntest,  測試集的row number
    # kf,     Kfold obj
    # train,  訓練集
    # labels, 目標
    # test    測試集
    
    # 先配置空間
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((fold, ntest)) # fold X ntest 空間 
    print("kf",kf)
    for i, (train_index, test_index) in enumerate(kf): # 開始分割
        print (train_index.max(),train.shape)
        x_tr = train[train_index]
        y_tr = labels[train_index]
        x_te = train[test_index]
        y_te = labels[test_index]

        model.train(x_tr, y_tr, x_te, y_te) # 訓練 (fold-1)個 fold

        oof_train[test_index] = model.predict(x_te) # 去預測 train left fold，稱作meta-train
        oof_test_skf[i, :] = model.predict(test) # 去預測 test，稱作meta-test

    oof_test[:] = oof_test_skf.mean(axis=0) # all folds score 取平均
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[411]:


class Xgb(object):
    def __init__(self, seed=2018, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 200) # 避免跑太久，所以設100

    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
            watchlist, verbose_eval=20)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

class Lgb(object):
    def __init__(self, seed=2018, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds',200)# 避免跑太久，所以設100

    def train(self, xtra, ytra, xte, yte):
        #ytra = ytra.ravel()
        #yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra)
        dvalid = lgb.Dataset(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(x)

class Cat(object):
    def __init__(self, seed=2018, params=None):
        self.seed = seed
        self.param = params
        self.nrounds = 200 # 避免跑太久，所以設100

    def train(self, xtra, ytra, xte, yte):
        self.gbdt = ctb.CatBoostRegressor(depth=14,
            iterations=self.nrounds, random_seed=self.seed,
            use_best_model=True)

        xtra = pd.DataFrame(xtra)
        ytra = pd.DataFrame(ytra)
        xte = pd.DataFrame(xte)
        yte = pd.DataFrame(yte)

        self.gbdt.fit(X=xtra, y=ytra, eval_set=(xte, yte),
                      use_best_model=True)

    def predict(self, x):
        return self.gbdt.predict(x)


# In[412]:


from sklearn.metrics import roc_auc_score


# In[413]:


def level_1(train, labels, test):
    #train = train
    #test = test
    #labels = labels

    ntrain = train.shape[0]
    ntest = test.shape[0]

    kf = KFold(ntrain, n_folds=fold ,
               shuffle=True, random_state=2018)

    lgb_params = {}
    lgb_params['boosting_type'] = 'gbdt'
    lgb_params['objective'] = 'binary'
    lgb_params['metric'] = 'binary_logloss'
    lgb_params['num_leaves'] = 2**5
    lgb_params['max_depth'] = 14
    lgb_params['feature_fraction'] = 0.9
    lgb_params['bagging_fraction'] = 0.95
    lgb_params['bagging_freq'] = 5
    lgb_params['learning_rate'] = 0.3

    xgb_params = {}
    xgb_params['booster'] = 'gbtree'
    xgb_params['objective'] = 'binary:logistic'
    xgb_params['learning_rate'] = 0.3
    xgb_params['max_depth'] = 14
    xgb_params['subsample'] = 0.8
    xgb_params['colsample_bytree'] = 0.7
    xgb_params['colsample_bylevel'] = 0.7

    cat_params = {}
    cat_params['learning_rate'] = 0.3
    cat_params['depth'] = 13
    cat_params['bagging_temperature'] = 0.8
    cat_params['loss_function']='AUC'
    cat_params['eval_metric']='AUC'
    
    cg = Cat(seed=2018, params=cat_params)
    xg = Xgb(seed=2018, params=xgb_params)
    lg = Lgb(seed=2018, params=lgb_params)
    
    ##########################################################################
    lg_oof_train, lg_oof_test = oof(lg, ntrain, ntest, kf, train, labels, test)
    xg_oof_train, xg_oof_test = oof(xg, ntrain, ntest, kf, train, labels, test)

    cg_oof_train, cg_oof_test = oof(cg, ntrain, ntest, kf, train, labels, test)
    ##########################################################################
    
    print("CG-CV: {}".format(mean_squared_error(labels, cg_oof_train)))
    print("XG-CV: {}".format(mean_squared_error(labels, xg_oof_train)))
    print("LG-CV: {}".format(mean_squared_error(labels, lg_oof_train)))

    x_train = np.concatenate((cg_oof_train, xg_oof_train, lg_oof_train), axis=1)
    x_test = np.concatenate((cg_oof_test, xg_oof_test, lg_oof_test), axis=1)

    np.save(arr=x_train, file='x_concat_train.npy')
    np.save(arr=x_test, file='x_concat_test.npy')
    np.save(arr=labels, file='y_labels.npy')

    return x_train, labels, x_test


# In[425]:


def level_2():
    train = np.load('x_concat_train.npy')
    labels = np.load('y_labels.npy')
    test = np.load('x_concat_test.npy')

    dtrain = xgb.DMatrix(train, label=labels)
    dtest = xgb.DMatrix(test)

    xgb_params = {}
    xgb_params["objective"] = "binary:logistic"
    xgb_params["eta"] = 0.1
    xgb_params["subsample"] = 0.9
    xgb_params["max_depth"] = 15
    xgb_params['eval_metric'] = 'auc'
    xgb_params['min_child_weight'] = 10
    xgb_params['seed'] = 2018

    res = xgb.cv(xgb_params, dtrain, num_boost_round=120, nfold=2, seed=2018, stratified=False,
                 early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('')
    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
    bst = xgb.train(xgb_params, dtrain, best_nrounds)

    preds = (bst.predict(dtest)) # 一開始把目標取了np.log1p()，現在inverse回去
    return preds


# In[415]:


# x_train.columns


# In[416]:


level_one=level_1(x_train.as_matrix(), y_train.as_matrix().reshape(-1),x_test.as_matrix())


# In[426]:


level_two=level_2()


# In[427]:


level_two


# In[423]:


Le2=pd.DataFrame(level_two)


# In[424]:


Le2.to_csv('level_two.csv')


# In[ ]:


# X_train


# In[121]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


x_train.head()


# In[123]:


x_train.info()


# In[ ]:


# PCA = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, loss=’hinge’, n_jobs=1, random_state=None, warm_start=False, class_weight=None, average=False, n_iter=None)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
Ridge = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)
SVC=LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000,  multi_class='ovr', penalty='l2', random_state=None, tol=0.001, verbose=0)
PCA.fit(X_train, Y_train,  eval_metric="auc",eval_set=eval_set, verbose=True)


# In[124]:


xgb=XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=120, silent=True, 
                  objective='binary:logistic', booster='gbtree', n_jobs=4, 
                  gamma=0, min_child_weight=5, max_delta_step=0, 
                  subsample=0.9, 
                  colsample_bytree=0.8, 
                  colsample_bylevel=0.8, 
                  reg_alpha=1, 
                  reg_lambda=0, 
                  scale_pos_weight=10, base_score=0.5, 
                  seed=42, missing=None)

# change seed number later on!!!!!!!!!!!!!!!!!!!!!!


# # This is for random search parameter till training step (not efficient tho, use grid serarch if possible)

# In[125]:


import numpy as np 
from time import time 
from scipy.stats import randint as sp_randint 
from sklearn.model_selection import RandomizedSearchCV 


# In[126]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[127]:


# specify parameters and distributions to sample from
param_dist = {"max_depth": sp_randint(10, 15), # "max_features": sp_randint(1, 11),
              "min_child_weight": sp_randint(3, 7)}

# min_child_weight = [1,10,50,100]
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]


# In[128]:


# run randomized search 
n_iter_search = 7
random_search = RandomizedSearchCV(xgb,  param_distributions=param_dist, 
n_iter=n_iter_search) 


# In[129]:


start = time() 
eval_set =  [(X_vad, Y_vad)]
random_search.fit(X_train, Y_train, eval_metric="auc",
          eval_set=eval_set, verbose=True)


# In[130]:


print("RandomizedSearchCV took %.2f seconds for %d candidates" 
" parameter settings." % ((time() - start), n_iter_search)) 
report(random_search.cv_results_)


# # This is for grid search till training step
# # Use this one more often

# In[ ]:


from sklearn.grid_search import GridSearchCV

# min_child_weight = [1,10,50,100]
# learning_rate = [0.01, 0.1]

para_grid = {'min_child_weight':[1, 10, 50, 100], 'learning_rate':[0.01, 0.1]}
svc = svm.SVC()
clf = GridSearchCV(xgb, param_grid=para_grid,cv=2)
clf.fit(iris.data, iris.target)

random_search = RandomizedSearchCV(xgb,  param_distributions=param_dist, 
n_iter=n_iter_search) 




eval_set =  [(X_vad, Y_vad)]

xgb=GridSearchCV(cv=2,estimator=xgb,param_grid={})


# In[131]:


Y_pred = xgb.predict_proba(X_vad)

Y_test2=xgb.predict_proba(x_test1)

pd.DataFrame(Y_test2).to_csv('Y_test2_0320_0.csv')


# In[128]:


# xgb=GridSearchCV(cv=2,estimator=xgb,param_grid={})


# In[110]:


eval_set =  [(X_vad, Y_vad)]
# model = XGBClassifier()
xgb.fit(X_train, Y_train,  eval_metric="auc",
          eval_set=eval_set, verbose=True)

Y_pred = xgb.predict_proba(X_vad)
# predict_proba
# accuracy = accuracy_score(Y_pred, Y_vad )
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(xgb.feature_importances_)

Y_test2=xgb.predict_proba(x_test1)

from xgboost import plot_importance
plot_importance(xgb)
plt.show()


# In[360]:


# print(xgb.feature_importances_)

Y_test2=xgb.predict_proba(x_test1)


# In[40]:


# Y_test2


# In[41]:


# x_test.head()


# In[ ]:



from xgboost import plot_importance
plot_importance(xgb)
plt.show()


# In[229]:


print( xgb.feature_importances_)


# In[361]:


sort_feature=xgb.feature_importances_
col_noF=f_log.columns[1:]
all_score=pd.DataFrame(sort_feature,col_noF,columns=['score'])


# In[232]:


all_score.to_csv('score_rank_point5_0314.csv')


# In[72]:


# A=np.random.random_integers(0,1,2,3,4)
# A


# In[73]:


# all_score[all_score['score']==0]


# In[148]:


# Train_dectected.groupby(["FileID","Day"])["Detected"].sum()


# In[74]:


import pickle
pickle.dump(xgb, open("xgboost__cus_pickle.dat", "wb"))


# In[117]:


loaded_model = pickle.load(open("xgboost_0948_sqrt_pickle.dat", "rb"))


# In[118]:


# loaded_model.feature_importances_


# In[362]:


pd.DataFrame(Y_test2).to_csv('Y_test2_0317_0.csv')


# # loaded_feature

# In[363]:


y_finish=pd.read_csv('Y_test2_0317_0.csv')


# In[364]:


x_tt=pd.read_csv('testing-set.csv',names=['FileID','score'])


# In[365]:


x_tt.head()


# In[366]:


y_finish['FileID']=x_tt['FileID']


# In[367]:


y_finish.drop(['Unnamed: 0'],axis=1).set_index('FileID').reset_index().to_csv('Y_test2_0314_1.csv')


# In[368]:


from xgboost import plot_importance
plot_importance(model)
plt.show()


# In[82]:


col_noF=f_log.columns[1:]


# In[83]:


all_score=pd.DataFrame(sort_feature,col_noF)
all_score.to_csv("0948_all_cus.csv")


# In[127]:


all_score.head()


# In[128]:


sort_feature


# In[372]:


from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC


# In[373]:


# PCA = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, loss=’hinge’, n_jobs=1, random_state=None, warm_start=False, class_weight=None, average=False, n_iter=None)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
Ridge = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)
SVC=LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000,  multi_class='ovr', penalty='l2', random_state=None, tol=0.001, verbose=0)
PCA.fit(X_train, Y_train,  eval_metric="auc",eval_set=eval_set, verbose=True)


# In[374]:


Ridge = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)


# In[115]:





# In[380]:


F=Ridge.score(X_train, Y_train)


# In[381]:


F


# In[376]:


SVC.fit(X_train, Y_train)


# In[ ]:


gb=GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’)


# In[ ]:


from mlxtend.classifier import StackingClassifier
meta_clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=1000, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
stacking_clf = StackingClassifier(classifiers=[PCA, Ridge, xbg, gb,PCA, Ridge, xbg, gb,PCA, Ridge, xbg, gb], meta_classifier=meta_clf)


# In[91]:



# importance = model.get_fscore
# importance = sorted(importance.items(), key=operator.itemgetter(1))


# In[ ]:


Train_data_df["FileID"].describe()


# In[136]:


# Train_data_df


# In[135]:


# Train_data_df.groupby(["FileID","QueryTs","Detected"]).sum()


# In[69]:



# for i in range(len(unix_time)):
#     local_time = time.localtime(unix_time[i])
#     TIME.append(time.strftime("%Y-%m-%d %H:%M:%S", local_time)) 
#     A,B=TIME[i].split(' ', 1 )
#     hour,miniute,second=B.split(":",2)
#     y,u,t=A.split("-",2)
#     day=u+t
#     Hour.append(hour)
#     Day.append(day)

np.array([[1,2,3,4,5],[4,5,6,7,8]])

# Data_log["Taiwan_time"]=pd.DataFrame(TIME)
# Data_log["Day"]=pd.DataFrame(Day)
# Data_log["Hour"]=pd.DataFrame(Hour)
# Data_log[["Taiwan_time"]].describe


# In[81]:


A=pd.DataFrame(np.array([[1,2,3],[1,1,0]]).T,columns=["first","third"])
B=pd.DataFrame(np.array([[1,2,3,3,5],[4,5,6,7,8]]).T,columns=["first","second"])
C=pd.merge(B,A,on='first')
C


# In[74]:








