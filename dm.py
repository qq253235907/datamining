import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import csv
from tqdm._tqdm import trange
import warnings
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')
#读取数据
age_train = pd.read_csv("age_train.csv", names=['uid','age_group'])
age_test = pd.read_csv("age_test.csv", names=['uid'])
user_basic_info = pd.read_csv("user_basic_info.csv", names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','fontSize','ct','carrier','os'])
user_behavior_info = pd.read_csv("user_behavior_info.csv", names=['uid','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'])
user_app_actived = pd.read_csv("user_app_actived.csv", names=['uid','appId'])
#user_app_usage = pd.read_csv("user_app_usage.csv")
app_info = pd.read_csv("app_info.csv", names=['appId', 'category'])

#处理数据量较大的user_app_usage.csv，结合app_info.csv简单统计得到appuseProcessed.csv作为特征
def f(x):
    s = x.value_counts()
    return np.nan if len(s) == 0 else s.index[0]
def processUserAppUsage():
    resTable = pd.DataFrame()
    reader = pd.read_csv("user_app_usage.csv", names=['uid','appId','duration','times','use_date'], iterator=True)
    last_df = pd.DataFrame()
    
    app_info = pd.read_csv("app_info.csv", names=['appId','category'])
    cats = list(set(app_info['category']))
    category2id = dict(zip(sorted(cats), range(0,len(cats))))
    id2category = dict(zip(range(0,len(cats)), sorted(cats)))
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])
    i = 1
    
    while True:
        try:
            print("index: {}".format(i))
            i+=1
            df = reader.get_chunk(1000000)
            df = pd.concat([last_df, df])
            idx = df.shape[0]-1
            last_user = df.iat[idx,0]
            while(df.iat[idx,0]==last_user):
                idx-=1
            last_df = df[idx+1:]
            df = df[:idx+1]

            now_df = pd.DataFrame()
            now_df['uid'] = df['uid'].unique()
            now_df = now_df.merge(df.groupby('uid')['appId'].count().to_frame(), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['appId','use_date'].agg(['nunique']), how='left', on='uid')
            now_df = now_df.merge(df.groupby('uid')['duration','times'].agg(['mean','max','std']), how='left', on='uid')    

            now_df.columns = ['uid','usage_cnt','usage_appid_cnt','usage_date_cnt','duration_mean','duration_max','duration_std','times_mean','times_max','times_std']


            df = df.merge(app_info, how='left', on='appId')
            now_df = now_df.merge(df.groupby('uid')['category'].nunique().to_frame(), how='left', on='uid')
            #print(df.groupby(['uid'])['category'].value_counts().index[0])
            now_df['usage_most_used_category'] = df.groupby(['uid'])['category'].transform(f)
            resTable = pd.concat([resTable, now_df])
        except StopIteration:
            break
    
    resTable.to_csv("appuseProcessed.csv",index=0)
    
    print("Iterator is stopped")

processUserAppUsage()

    #将user_basic_info.csv 和 user_behavior_info.csv中的字符值编码成可以训练的数值类型，合并
class2id = {}
id2class = {}
def mergeBasicTables(baseTable):
    resTable = baseTable.merge(user_basic_info, how='left', on='uid', suffixes=('_base0', '_ubaf'))
    resTable = resTable.merge(user_behavior_info, how='left', on='uid', suffixes=('_base1', '_ubef'))
    cat_columns = ['city','prodName','color','carrier','os','ct']
    for c in cat_columns:
        resTable[c] = resTable[c].apply(lambda x: x if type(x)==str else str(x))
        sort_temp = sorted(list(set(resTable[c])))  
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        id2class['id2'+c] = dict(zip(range(1,len(sort_temp)+1), sort_temp))
        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])
        
    return resTable

#处理app使用相关数据
#对user_app_actived.csv简单统计
#将之前训练的appuseProcess.csv进行合并

# 进阶——对user_app_actived.csv高级统计——统计用户安装的每种APP的数目

appuseProcessed_df = pd.read_csv("appuseProcessed.csv")

def mergeAppDataUpgrade(baseTable):

    appusedTable = pd.read_csv("appuseProcessed.csv")
    resTable = baseTable.merge(user_app_actived, how='left', on='uid')
    resTable['appId'] = resTable['appId'].apply(lambda x: len(list(x.split('#'))))

    resTable = resTable.merge(appusedTable, how='left', on='uid')
    resTable[['category', 'usage_most_used_category']] = resTable[['category', 'usage_most_used_category']].fillna(41)
    resTable = resTable.fillna(0)


    now_df = pd.DataFrame()
    now_df['uid'] = resTable['uid']
    for i in range(40):
        now_df[str(i)] = 0

    now_df = pd.merge(resTable, now_df, how='left', on='uid')

    now_df = now_df.fillna(0)



    # 注意给的数据中，用户激活的app中有一些是不在app_info 中的，可以先忽略或者单独归为一类
    app_info = pd.read_csv("app_info.csv", names=['appId','category'])
    # 统计出共有多少种类别的app
    cats = list(set(app_info['category']))
    category2id = dict(zip(sorted(cats), range(0,len(cats))))
    id2category = dict(zip(range(0,len(cats)), sorted(cats)))
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])
    
    app_info_dict = app_info.set_index('appId').T.to_dict('list')

    now_df = pd.merge(now_df, user_app_actived, how='left', on='uid')
    
    try:
        with tqdm(range(0, len(now_df))) as t:
            for index in t:
                appIdList = now_df.at[index, 'appId_y'].split('#')
                for appId in appIdList:
                    if appId in app_info_dict.keys():
                        now_df.at[index, str(app_info_dict[appId][0])] += 1
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()     

    now_df = now_df.drop(['appId_y'], axis=1)
   
    return now_df
#合并用户基本特征以及app使用相关特征，作为训练集和测试集
df_train = mergeAppDataUpgrade(mergeBasicTables(age_train))
df_test = mergeAppDataUpgrade(mergeBasicTables(age_test))
df_train.to_csv("train_features.csv",index=0)
df_test.to_csv("test_features.csv",index=0)
#df_train = mergeAppData(mergeBasicTables(age_train))
#df_test = mergeAppData(mergeBasicTables(age_test))
print(df_train.shape)
print(df_test.shape)

#训练模型

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

print("训练模型：")
param = {
        'learning_rate': 0.12,
        'lambda_l1': 0.1,
        'lambda_l2': 0.02,
        'max_depth': 20,
        'objective': 'multiclass',
        'num_class': 7,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'max_bin': 230,
        'feature_fraction': 0.8,
        'metric': 'multi_error'
        }

X = df_train.drop(['age_group','uid'], axis=1)
y = df_train['age_group']
uid = df_test['uid']
test = df_test.drop('uid', axis=1)

xx_score = []
cv_pred = []
skf = StratifiedKFold(n_splits=3, random_state=1030, shuffle=True)
for index, (train_index, vali_index) in enumerate(skf.split(X, y)):
    print(index)
    x_train, y_train, x_vali, y_vali = np.array(X)[train_index], np.array(y)[train_index], np.array(X)[vali_index], np.array(y)[vali_index]
    train = lgb.Dataset(x_train, y_train)
    vali =lgb.Dataset(x_vali, y_vali)
    print("training start...")
    model = lgb.train(param, train, num_boost_round=1000, valid_sets=[vali], early_stopping_rounds=50)
    xx_pred = model.predict(x_vali,num_iteration=model.best_iteration)
    xx_pred = [np.argmax(x) for x in xx_pred]
    xx_score.append(f1_score(y_vali,xx_pred,average='weighted'))
    y_test = model.predict(test,num_iteration=model.best_iteration)
    y_test = [np.argmax(x) for x in y_test]
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
        
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
df = pd.DataFrame({'id':uid.as_matrix(),'label':submit})
df.to_csv('submission.csv',index=False)