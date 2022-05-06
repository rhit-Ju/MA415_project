#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('C:/Users/zhangy30/Desktop/Data_MA415_local/high_diamond_ranked_10min.csv')
df.head(3)


# In[3]:


df = df.drop(df.columns[0], axis = 1)    # drop the 'gameId' column

df = df[df['blueWardsPlaced'] <= 50]  # drop the edge cases, wards placed > 50
df = df[df['redWardsPlaced'] <= 50]

df = df.drop(['blueEliteMonsters', 'blueGoldPerMin', 'redFirstBlood', 'redKills',
              'redDeaths', 'redEliteMonsters', 'redGoldDiff', 'redGoldPerMin'], axis = 1)
df.shape


# In[4]:


X = df.drop('blueWins', axis = 1)
y = df.blueWins

X = (X - X.mean())/X.std()


# ## Logistic Only

# In[5]:


lgr = LogisticRegression()

scores = cross_validate(lgr, X, y, return_train_score = True)
R2_train = scores['train_score'].mean()
R2_valid = scores['test_score'].mean()

print('train R2', R2_train.round(3))
print('validation R2', R2_valid.round(3))


# # Logistic Regressor and Feature Engineering

# In[8]:



degree = []
train_R2 = []
valid_R2 = []

for k in np.arange(3) + 1:
    lgr = LogisticRegression(fit_intercept=False)
    poly = PolynomialFeatures(k, interaction_only=False, include_bias=True)
    Xe = poly.fit_transform(X)
    degree.append(k)
    cros = cross_validate(lgr, Xe, y, return_train_score=True)
    train_R2.append(cros['train_score'].mean())
    valid_R2.append(cros['test_score'].mean())
#     print(Xe.shape)

results = pd.DataFrame()
results['degree'] = degree
results['train R2'] = train_R2
results['valid R2'] = valid_R2

print(results)


# In[9]:


print("Looks like a degree of 2 is better, 3 is a little overfitting")


# ## PCA

# In[11]:


# %load "~/Desktop/Data_MA415_local/biplot.py
def biplot(X,n_comp,j1,j2,scale=1,alpha=0.25,s=50):
    
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    #X = (X-X.mean())/X.std()
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    cols = ['PC-'+str(i+1) for i in range(n_comp)]
    Z = pca.transform(X)
    Z = pd.DataFrame(Z,columns=cols)
    Z.plot.scatter(x=j1-1,y=j2-1,
                   alpha=alpha,
                   s=s,
                   figsize=(10,10))
    sd = np.sqrt(pca.explained_variance_)
    Zj1 = sd[j1-1]*np.array([1,0])
    Zj2 = sd[j2-1]*np.array([0,1])
    plt.arrow(0,0,Zj1[0],Zj1[1],head_width=0.1,color='k')
    plt.arrow(0,0,Zj2[0],Zj2[1],head_width=0.1,color='k')

    L = pca.components_
    L = pd.DataFrame(L,index=cols,columns=X.columns)
    for k in range(X.shape[1]):
        x = scale*L.iloc[j1-1,k]
        y = scale*L.iloc[j2-1,k]
        plt.arrow(0,0,x,y,head_width=0.1,color='r')
        plt.text(1.2*x,1.2*y,X.columns[k],color='black')
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        
    PVE = pca.explained_variance_ratio_
    PVE = pd.Series(PVE,index=cols)
    return PVE


# In[15]:


PVE = biplot(X, 30, 1, 2, scale=2.5, alpha = 0.5, s = 100)


# In[16]:


PVE


# In[17]:


PVE.plot.barh(color='cyan')


# ## Random Forest

# In[21]:


rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(X, y)
print(f'out of bag R2 {rf.oob_score_.round(3)}')


# ## Random Forest with Feature Engineering

# In[23]:


poly = PolynomialFeatures(2, interaction_only=False, include_bias=True)
Xe = poly.fit_transform(X)

rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf.fit(Xe, y)
print(f'out of bag R2 {rf.oob_score_.round(3)}')


# In[ ]:




