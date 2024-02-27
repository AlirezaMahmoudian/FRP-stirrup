#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer 
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars ,RANSACRegressor, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager
import random
from lazypredict.Supervised import LazyRegressor
from sklearn.svm import SVR


# In[ ]:


df = pd.read_excel(r'D:\Articles\FRP stirrup\New\Dataset.xlsx',sheet_name='Dataset' ,header = 0 )
y = df.iloc[:, 5].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4]].to_numpy()
df.tail(2)


# In[ ]:


Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.7 ,random_state=0 )
model=KNeighborsRegressor()
model.fit(Xtr , ytr)
model.fit(Xtr , ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)
r2tr=round(r2_score(ytr , yprtr),2)
r2te=round(r2_score(yte , yprte),2)
msetr=round(mean_squared_error(ytr , yprtr)**0.5,2)
msete=round(mean_squared_error(yte , yprte)**0.5,2)
maetr= round(mean_absolute_error(ytr , yprtr),2)
maete= round(mean_absolute_error(yte , yprte),2)
a = min([np.min(yprtr), np.min(yprte)])
b = max([np.max(yprtr), np.max(yprte), 1])
plt.scatter(ytr, yprtr, s=80, facecolors='blue', edgecolors='black',
            label=f'\n Train \n R2 = {r2tr}  \nRMSE = {msetr}\nMAE = {maetr}')
plt.scatter(yte , yprte,s=80, marker='s',facecolors='firebrick', edgecolors='black',
           label=f'\n Test \n R2 = {r2te} \nRMSE = {msete}\nMAE = {maete}')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'KNeighbors Regressor ',fontsize=14)
plt.xlabel('Fu [MPa]_Experimental',fontsize=15)
plt.ylabel('Fu [MPa]_Predicted',fontsize=15)
# Customizing the text font and size
font = {'family': 'Times New Roman', 'size': 12}
plt.rc('font', **font)
plt.legend(loc=4)
plt.tight_layout()
plt.savefig(r"D:\Articles\FRP stirrup\New\New figs\KNN1.jpg", dpi=1000,format='jpeg')
plt.show()
print(sqrt(mean_squared_error(yte, yprte)))


# In[ ]:


import matplotlib.pyplot as plt

df.hist(bins=20, grid=False, figsize=(12, 8), color='red')  # Change the color here
plt.savefig(r"D:\Articles\FRP stirrup\New\New figs\hist.SVG", dpi=1000, format='SVG')
plt.show()

