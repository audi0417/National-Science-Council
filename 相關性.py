#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import requests
import json
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


def Get_world_Stock():
    url = "https://finance.yahoo.com/world-indices/"
    response = requests.get(url)

    import io
    f = io.StringIO(response.text)
    dfs = pd.read_html(f)
    return dfs[0]

def Stock_price(stock_id):
    start = datetime.datetime(2010,1,1)
    end = datetime.datetime(2018,12,31)
    df = web.DataReader(stock_id, "yahoo", start, end)
   
    return df


world_index = Get_world_Stock()


# In[64]:


world_index


# In[65]:


world_Stock_index = {}

for symbol, name in zip(world_index['Symbol'], world_index['Name']):
    try:
        world_Stock_index[name] = Stock_price(symbol)
    except :
        print(name)


# In[66]:


world_Stock_index


# In[67]:


#利用開盤價建立表格
Close = {}
for name, price in world_Stock_index.items():
    if len(price) != 0:
        Close[name] = world_Stock_index[name]['Close']


# In[68]:


Close


# In[69]:


Close = pd.DataFrame(Close)

Close = Close.resample('1d').last()
Close=Close.drop({'Cboe UK 100','S&P/CLX IPSA','Jakarta Composite Index','IPC MEXICO'
    ,'HANG SENG INDEX','IBOVESPA','NYSE AMEX COMPOSITE INDEX','BEL 20','Top 40 USD Net TRI Index'
           ,'CAC 40','ESTX 50 PR.EUR',},axis=1)

Close_corr=Close.corr()


# In[70]:


Close_corr


# In[71]:


#利用視覺化套件 seaborn 繪製熱力圖
import matplotlib.pyplot as plt
import seaborn 
plt.rcParams['figure.figsize'] = (14, 14)

seaborn.heatmap(Close_corr,annot=True,cmap="spring",linewidths=.3)


# In[72]:


import scipy


# In[49]:


df = pd.read_csv("/Users/chenkaijung/Data_x3/TW.csv")
df1 = pd.read_csv("/Users/chenkaijung/Data_x3/Sp500.csv")
df2 = pd.read_csv("/Users/chenkaijung/Data_x3/NASDAQ.csv")


# In[50]:


scipy.stats.levene(df['Close'],df1['Close'])


# In[51]:


print(scipy.stats.pearsonr(df['Close'],df1['Close']))
print(scipy.stats.spearmanr(df['Close'],df1['Close']))
print(scipy.stats.kendalltau(df['Close'],df1['Close']))


# In[52]:


print(scipy.stats.pearsonr(df2['Close'],df1['Close']))
print(scipy.stats.spearmanr(df2['Close'],df1['Close']))
print(scipy.stats.kendalltau(df2['Close'],df1['Close']))


# In[ ]:




