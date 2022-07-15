#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black', labelsize=20)
plt.rc('ytick', direction='out', color='black', labelsize=20)
font = {'family' : 'normal',
        'size'   : 20}
plt.rc('font', **font)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


# In[8]:


u = pd.read_csv('../LCPB/seq_len.csv')
u.head()


# In[9]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['seq_len']
y1 = u['TL']
# plotting the line 1 points
plt.plot(x1, y1, label = "train loss")

# line 2 points
x2 = u['seq_len']
y2 = u['VL']
# plotting the line 2 points
plt.plot(x2, y2, label = "validation loss")

# line 3 points
x3 = u['seq_len']
y3 = u['TL.1']
# plotting the line 3 points
plt.plot(x3, y3, label = "test loss")


# naming the x axis
plt.xlabel('sequence length')
# naming the y axis
plt.ylabel('loss')
# giving a title to  graph
plt.title('sequence length vs loss')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()


# In[ ]:





# In[11]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['seq_len']
y1 = u['TA']
err1 = [0] * 6
# plotting the line 1 points
plt.plot(x1, y1, label = "train accuracy")
# Plot error bar
plt.fill_between(x1, y1-err1, y1+err1)

# line 2 points
x2 = u['seq_len']
y2 = u['VA']
err2 = [0.03] * 6
# plotting the line 2 points
plt.plot(x2, y2, label = "validation accuracy")
# Plot error bar
plt.fill_between(x2, y2-err2, y2+err2, alpha=0.2)

# line 3 points
x3 = u['seq_len']
y3 = u['TA.1']
err3 = [0.03] * 6
# plotting the line 3 points
plt.plot(x3, y3, label = "test accuracy")
# Plot error bar  
plt.fill_between(x3, y3-err3, y3+err3, alpha=0.2)
# naming the x axis
plt.xlabel('sequence length')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to  graph
plt.title('sequence length vs accuracy')

# show a legend on the plot
plt.legend()



# function to show the plot
plt.show()


# In[12]:


u = pd.read_csv('../LCPB/full_protein_input.csv')
u.head()


# In[13]:


df= pd.read_csv('../LCPB/one_aminoacid_input.csv')
df.head()


# In[14]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['EP']
y1 = u['TL']
# plotting the line 1 points
plt.plot(x1, y1, label = "train loss full protein")

# line 2 points
x2 = u['EP']
y2 = u['VL']
# plotting the line 2 points
plt.plot(x2, y2, label = "validation loss full protein")

# line 3 points
x3 = u['EP']
y3 = u['TEL']
# plotting the line 3 points
plt.plot(x3, y3, label = "test loss full protein")


# naming the x axis
plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('loss')
# giving a title to  graph
plt.title('Number of Epochs vs loss ')


########

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x4 = df['EP']
y4 = df['TL']
# plotting the line 1 points
plt.plot(x4, y4, label = "train loss one protein",linestyle='dashed', linewidth = 3)


# line 2 points
x5 = df['EP']
y5 = df['VL']
# plotting the line 2 points
plt.plot(x5, y5, label = "validation loss one protein",linestyle='dashed', linewidth = 3)


# line 3 points
x6 = df['EP']
y6 = df['TEL']
# plotting the line 3 points
plt.plot(x6, y6, label = "test loss one protein",linestyle='dashed', linewidth = 3)


# naming the x axis
#plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('loss')
# giving a title to  graph
#plt.title('Number of Epochs vs loss ')

# show a legend on the plot
plt.legend()

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# function to show the plot
plt.show()

  


# In[15]:


u = pd.read_csv('../LCPB/full_protein_input.csv')
u.head()


# In[17]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['EP']
y1 = u['TA']
# plotting the line 1 points
plt.plot(x1, y1, label = "train accuracy full protein")

# line 2 points
x2 = u['EP']
y2 = u['VA']
# plotting the line 2 points
plt.plot(x2, y2, label = "validation accuracy full protein")

# line 3 points
x3 = u['EP']
y3 = u['TEA']
# plotting the line 3 points
plt.plot(x3, y3, label = "test accuracy full protein")


# naming the x axis
plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('Accuracy')
# giving a title to  graph
plt.title('Number of Epochs vs  accuracy ')

# show a legend on the plot
#plt.legend()

# function to show the plot
#plt.show()




########

plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x4 = df['EP']
y4 = df['TA']
# plotting the line 1 points
plt.plot(x4, y4, label = "train  accuracy one aminoacid",linestyle='dashed', linewidth = 3)


# line 2 points
x5 = df['EP']
y5 = df['VA']
# plotting the line 2 points
plt.plot(x5, y5, label = "validation accuracy one aminoacid",linestyle='dashed', linewidth = 3)


# line 3 points
x6 = df['EP']
y6 = df['TEA']
# plotting the line 3 points
plt.plot(x6, y6, label = "test  accuracy one aminoacid",linestyle='dashed', linewidth = 3)


# naming the x axis
#plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel(' accuracy')
# giving a title to  graph
#plt.title('Number of Epochs vs loss ')

# show a legend on the plot
plt.legend()

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# function to show the plot
plt.show()


# In[18]:


u = pd.read_csv('../LCPB/window.csv')
u.head()


# In[19]:


plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['window']
y1 = u['train_loss']
# plotting the line 1 points
plt.plot(x1, y1, label = "train loss")

# line 2 points
x2 = u['window']
y2 = u['valid_loss']
# plotting the line 2 points
plt.plot(x2, y2, label = "validation loss")

# line 3 points
x3 = u['window']
y3 = u['test_loss']
# plotting the line 3 points
plt.plot(x3, y3, label = "test loss")


# naming the x axis
plt.xlabel(' window size')
# naming the y axis
plt.ylabel('loss')
# giving a title to  graph
plt.title('window size vs loss')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()


# In[20]:


u = pd.read_csv('../LCPB/window.csv')
u.head()


# In[21]:


plt.rcParams["figure.figsize"] = (12,6)
# line 1 points
x1 = u['window']
y1 = u['train_accuracy']
err1 = u['train_std']
# plotting the line 1 points
plt.plot(x1, y1, label = "train aacuracy")
# Plot error bar
plt.fill_between(x1, y1-err1, y1+err1, alpha=0.2)

# line 2 points
x2 = u['window']
y2 = u['valid_accuracy']
err2 = u['val_std']
# plotting the line 2 points
plt.plot(x2, y2, label = "validation accuracy")
# Plot error bar
plt.fill_between(x2, y2-err2, y2+err2, alpha=0.2)

# line 3 points
x3 = u['window']
y3 = u['test_accuracy']
err3 = u['test_std']
# plotting the line 3 points
plt.plot(x3, y3, label = "test accuracy")
# Plot error bar
plt.fill_between(x3, y3-err3, y3+err3, alpha=0.2)


# naming the x axis
plt.xlabel(' window size')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to  graph
plt.title('window size vs accuracy')

# show a legend on the plot
plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# function to show the plot
plt.show()



# In[22]:


u = pd.read_csv('../LCPB/cath_seq.csv')
u.head()


# In[23]:


df= pd.read_csv('../LCPB/pdb_seq.csv')
df.head()


# In[24]:


plt.rcParams["figure.figsize"] = (12,6)

# line 1 points
x1 = u['seq_len']
y1 = u['test_accuracy']
err1 = u['test_std']
# plotting the line 3 points
plt.plot(x1, y1, label = "test accuracy of cath")
# Plot error bar
plt.fill_between(x1, y1-err1, y1+err1, alpha=0.2)


# line 2 points
x2 = df['seq_len']
y2 = df['test_accuracy']
err2 = df['test_std']
# plotting the line 2 points
plt.plot(x2, y2, label = "test accuracy of pdb")
# Plot error bar
plt.fill_between(x2, y2-err2, y2+err2, alpha=0.2)


# naming the x axis
plt.xlabel(' sequence length')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to  graph
plt.title('sequence length vs accuracy in pdb and cath datasete')

# show a legend on the plot
plt.legend()
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# function to show the plot
plt.show()


# In[25]:


u = pd.read_csv('../LCPB/821039_ltc_32_10_Adam_epochs_NNI.csv')
u.head()


# In[26]:


df= pd.read_csv('../LCPB/822485_ltc_32_10_Adam_epochs_NNIH.csv')
df.head()


# In[28]:


plt.rcParams["figure.figsize"] = (12,6)

# line 1 points
x1 = u['EP']
y1 = u['TEA']
err1 = u['TESTD']
# plotting the line 3 points
plt.plot(x1, y1, label = "test accuracy in NNI")
# Plot error bar
plt.fill_between(x1, y1-err1, y1+err1, alpha=0.2)


# line 2 points
x2 = df['EP']
y2 = df['TEA']
err2 = df['TESTD']
# plotting the line 2 points
plt.plot(x2, y2, label = "test accuracy in NNIH")
# Plot error bar
plt.fill_between(x2, y2-err2, y2+err2, alpha=0.2)


# naming the x axis
plt.xlabel(' Number of epochs')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to  graph
plt.title('Number of epochs vs accuracy in NNI and NNIH')

# show a legend on the plot
plt.legend()
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# function to show the plot
plt.show()

