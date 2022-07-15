#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


u = pd.read_csv('../LCPB/2018-06-06-pdb-intersect-pisces.csv')


# In[3]:


u.head()


# In[4]:




df2 = u[["pdb_id",'seq', 'sst8',"len"]]
df2.head()


# In[5]:


df2["len"]


# In[9]:


g=open("data","w")
g.write("")
for j in range(0,9078):
    g.write("#"+df2["pdb_id"][j]+"\n")
    for i in range(0,df2["len"][j]):
        g.write(df2["seq"][j][i]+" "+df2["sst8"][j][i]+"\n")
g.close()

