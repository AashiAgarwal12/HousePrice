#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# In[2]:


# Common imports
import numpy as np
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[3]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[4]:


fetch_housing_data()


# In[5]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[6]:


housing = load_housing_data()
housing.head()


# In[7]:


housing.info()


# In[8]:


housing.describe()


# In[9]:


housing["ocean_proximity"].value_counts()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()


# In[11]:


y = housing['median_income']


# In[12]:


y


# In[13]:


X = housing.drop(columns=['median_income'])


# In[14]:


X


# In[15]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[16]:


# X_train, X_test, y_train, y_test = train_test_split(
# ...     X, y, test_size=0.33, random_state=42)


# In[17]:


corr_matrix = housing.corr()


# In[18]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[19]:


# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")


# In[20]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[21]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[22]:


housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[23]:


housing.describe()


# In[24]:


housing = housing.dropna(subset=["total_bedrooms"]) 


# In[25]:


# housing.drop("total_bedrooms", axis=1)      


# In[26]:


housing.shape


# In[27]:


housing.drop(['rooms_per_household','bedrooms_per_room','population_per_household'],axis = 1)


# In[28]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[29]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[31]:


housing['ocean_proximity'] = ordinal_encoder.fit_transform(housing[['ocean_proximity']])


# In[32]:


housing


# In[33]:


X = housing.drop(['median_house_value','rooms_per_household','bedrooms_per_room','population_per_household'],axis=1)
Y= housing['median_house_value']


# In[34]:


X


# In[35]:


Y


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


reg = LinearRegression().fit(X_train, Y_train)
reg.score(X_train,Y_train)


# In[39]:


reg.coef_


# In[40]:


reg.intercept_


# In[41]:


reg = LinearRegression().fit(X_test, Y_test)
reg.score(X_test,Y_test)


# In[ ]:




