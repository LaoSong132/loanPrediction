#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[4]:


import numpy as np 
import pandas as pd 

#EDA 
import seaborn as sns 
import matplotlib.pyplot as plt

#Imputation 
from sklearn.impute import SimpleImputer

#split

from sklearn.model_selection import train_test_split

# Deep Learning 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Evaluation
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve


# # Read Data and review

# In[5]:


df = pd.read_csv('data/accepted_2007_to_2018Q4.csv',
                 usecols=['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
                          'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
                          'verification_status', 'issue_d', 'loan_status', 'purpose', 'title',"addr_state",
                          'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
                          'revol_util', 'total_acc', 'initial_list_status', 'application_type',
                          'mort_acc', 'pub_rec_bankruptcies'])


# note the null values in target 

# # Target Preprocessing
# Looking at the target there are multiple categories. \
# For our analysis we want to define a Binary classification of ***Default vs Paid***

# In[7]:


replace_status = {"Fully Paid":"Paid",
             "Current": "Paid",
             "Charged Off": "Default",
              "Does not meet the credit policy. Status:Charged Off":"Default",
              "Does not meet the credit policy. Status:Fully Paid":"Paid",
              "Late (31-120 days)":"Late",
              "Late (16-30 days)":"Late",
              "In Grace Period":"Late",
              "Default":"Default"
             }


# In[8]:


df["loan_status"] = df["loan_status"].replace(replace_status)


# We will drop everything NOT ***Default or Paid*** i.e. null and Late \
# In another notebook we can investigate how ***Late*** payments affect ***Default*** and look at imputing the null values

# In[9]:


# Keep Default or Paid loans only
df = df[ (df["loan_status"]== "Paid") | (df["loan_status"]== "Default")]



# In[9]:


def create_countplot(axes, x_val,order_val, title, rotation="n"):
    sns.countplot(ax= axes, data=df, x=x_val, order = order_val.value_counts(dropna= False).index,hue = "loan_status")
    axes.set_title(title)
    if rotation =="y":
        axes.set_xticklabels(list(order_val.unique()), rotation=90)


with plt.style.context(['science','ieee','no-latex']):
    fig, ax = plt.subplots(2,3, figsize= (20,10))

    create_countplot(ax[0,0],'term', df["term"],"The number of payments on the loan (months)" )

    create_countplot(ax[0,1],'grade', df["grade"],"Loan grade")

    create_countplot(ax[0,2],'sub_grade', df["sub_grade"],"Loan sub_grade","y")

    create_countplot(ax[1,0],'emp_length', df["emp_length"],"Borrower length of employment (years)", "y" )

    create_countplot(ax[1,1],'home_ownership', df["home_ownership"],"Borrower home ownership status" )

    create_countplot(ax[1,2],'verification_status', df["verification_status"],"verification_status" )


    plt.tight_layout()
    plt.show()

## too many unique titles to plot 
df["emp_title"].value_counts(dropna= False)




# Puropose and Title are essentially duplicates with Purpose being more descriptive \
# As such we can drop Title 

# ### Date Analysis


#convert to date 
df["issue_d"] = pd.to_datetime(df["issue_d"])
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])


# ### 2. Correlation Analysis

with plt.style.context(['science','ieee','no-latex']):
    plt.figure(figsize= (15,7))
    sns.heatmap(df.corr(), vmin=1, vmax=-1, annot=True, cmap="Spectral")

imputer_mean = SimpleImputer() #mean imputation
imputer_mode = SimpleImputer(strategy="most_frequent")

## Reset index for concat 
df = df.reset_index(drop = True)


mode_impute = ["emp_title","earliest_cr_line"]
mean_impute = ["annual_inc","dti","open_acc","pub_rec","revol_util","total_acc","mort_acc","pub_rec_bankruptcies"]

mean_df = pd.DataFrame(data = imputer_mean.fit_transform(df[mean_impute]), columns = mean_impute)

df.drop(mean_impute,axis = 1,inplace =True)

df = pd.concat([df,mean_df],axis =1)

df.head()

df["emp_length"].fillna(df["emp_length"].mode()[0], inplace = True)
df["earliest_cr_line"].fillna(df["earliest_cr_line"].mode()[0],inplace = True)

# ## Drop 
# * emp_title
# * title
# * grade

# too many unique values 
df.drop("emp_title",axis =1, inplace = True)

# title is the same as "purpose" we can therefore drop this column
df.drop("title",axis =1, inplace = True)

## grade holds the same information as subgrade
df.drop("grade",axis =1, inplace = True)

df["term"] = df["term"].apply(lambda x : x[:3]).astype(int)
print(df["term"].value_counts())

df["emp_length"].value_counts()


replace_dictionary = {"< 1 year":"1 years" }
df["emp_length"].replace(replace_dictionary,inplace=True)


df["emp_length"] =df["emp_length"].apply(lambda x: x[:2]).astype(int)


# ## Dummies
# Convert all categorical (non-ordinal) features into dummy columns including the target column

## Target to dummies 
df["loan_status"] = df["loan_status"].map({"Paid":0,"Default":1})

df["home_ownership"].value_counts()

# lets group None and Any --> Other
df["home_ownership"]= df["home_ownership"].replace(["ANY","NONE"], "OTHER")

dummy_cols = [ "home_ownership", "verification_status", "purpose","initial_list_status", "application_type","sub_grade", "addr_state"]

#get dummy columns
df_dummies = pd.get_dummies(df[dummy_cols], drop_first=True)

#drop from original dataframe
df.drop(dummy_cols,axis =1, inplace=True)

df= pd.concat([df,df_dummies],axis =1)

df.drop("issue_d",axis =1, inplace=True)

print(df["earliest_cr_line"].value_counts())


# We can ignore the day value as this is only the first
# extract year column 
df["year_earliest"] = pd.to_datetime(df["earliest_cr_line"]).dt.year

#extract month column 
df["month_earliest"] = pd.to_datetime(df["earliest_cr_line"]).dt.month

#drop old column as we dont it now
df.drop(["earliest_cr_line"],axis=1, inplace=True)

df.iloc[:,:20].info()

# # Split

df["loan_status"].value_counts()

X = df.drop("loan_status",axis =1 )
y= df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Scaling
from sklearn.preprocessing import MinMaxScaler


# In[36]:


scaler = MinMaxScaler()


# In[37]:


# we only transform X_test to stop any leakage 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





# # XGBoost RandomForest Logistic regression


from xgboost import XGBClassifier
xgbc=XGBClassifier(scale_pos_weight=10)
param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,6,7,8,9,10,11,12,13,14],
              'n_estimators': [50,65,80,100,115,130,150],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
clf = BayesSearchCV(estimator=xgbc, search_spaces=param_grid,cv=3, return_train_score=True, verbose=3)
clf.fit(X_train, y_train)

#results dataframe
df_test = pd.DataFrame(clf.cv_results_)
    
#predictions - inputs to confusion matrix
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
unseen_predictions = clf.predict(df_test.iloc[:,1:])
    
#confusion matrices
cfm_train = confusion_matrix(y_train, train_predictions)
cfm_test = confusion_matrix(y_test, test_predictions)
cfm_unseen = confusion_matrix(df_test.iloc[:,:1], unseen_predictions)
    
#accuracy scores
accs_train = accuracy_score(y_train, train_predictions)
accs_test = accuracy_score(y_test, test_predictions)
accs_unseen = accuracy_score(df_test.iloc[:,:1], unseen_predictions)
    
#F1 scores for each train/test label
f1s_train_p1 = f1_score(y_train, train_predictions, pos_label=1)
f1s_train_p0 = f1_score(y_train, train_predictions, pos_label=0)
f1s_test_p1 = f1_score(y_test, test_predictions, pos_label=1)
f1s_test_p0 = f1_score(y_test, test_predictions, pos_label=0)
    
#Area Under the Receiver Operating Characteristic Curve
test_ras = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
unseen_ras = roc_auc_score(df_test.iloc[:,:1], clf.predict_proba(df_test.iloc[:,1:])[:,1])
    
#best parameters
bp = clf.best_params_
    
#storing computed values in results dictionary
results_dict = {'classifier': deepcopy(clf),
                            'cv_results': df.copy(),
                            'cfm_train': cfm_train,
                            'cfm_test': cfm_test,
                            'cfm_unseen': cfm_unseen,
                            'train_accuracy': accs_train,
                            'test_accuracy': accs_test,
                            'unseen_accuracy': accs_unseen,
                            'train F1-score label 1': f1s_train_p1,
                            'train F1-score label 0': f1s_train_p0,
                            'test F1-score label 1': f1s_test_p1,
                            'test F1-score label 0': f1s_test_p0,
                            'unseen F1-score label 1': f1s_unseen_p1,
                            'unseen F1-score label 0': f1s_unseen_p0,
                            'test roc auc score': test_ras,
                            'unseen roc auc score': unseen_ras,
                            'best_params': bp}
with plt.style.context(['science','ieee','no-latex']):
    plot_roc_curve(clf,X_test,y_test)

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

clf2 =LogisticRegression(random_state=0,class_weight='balanced',penalty='elasticnet',solver='saga',l1_ratio=0.1,max_iter=10000)   
clf2.fit(X_res,y_res)
with plt.style.context(['science','ieee','no-latex']):
    plot_roc_curve(clf2,X_test,y_test)

plot_roc_curve(clf1, X_test, y_test) 

print(classification_report(y_test,clf.predict(X_test)))

