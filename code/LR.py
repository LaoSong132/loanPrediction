#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)

# ignore the warning if any
import warnings
warnings.filterwarnings("ignore")

# set row/columns
pd.options.display.max_columns= None
pd.options.display.max_rows= None
np.set_printoptions(suppress=True)


# In[2]:


data = pd.read_csv('data/accepted_2007_to_2018Q4.csv')


# In[3]:


data.head(5)


# In[4]:


data.shape


# In[5]:


# check the data frame information
data.info(verbose= True)


# In[6]:


# Generate descriptive statistics, so as to get the overall behaviour of different columns
# i.e. its mean, standard deviation, minimum/maximum value ,
# Descriptive statistics include those that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.
#For numeric data, the result’s index will include count, mean, std, min, max as well as lower,
#50 and upper percentiles. By default the lower percentile is 25 and the upper percentile is 75. 
#The 50 percentile is the same as the median.
data.describe()


# In[7]:


# check the number of NaN values (also known as Missing Values) present in each column
data.isnull().sum()


# In[8]:


# Make a copy of df, so that we can apply all the operation on df_copy without modifying the content of df
data_copy = data.copy()


# In[9]:


# drop all the NaN values, to see is it contains any rows after deleting all NaN values.
data_copy.dropna()


# - Since it is returing empty dataframe , we need to consider other preprocessing/ treatment on data to get rid of NaN (Missing) value

# In[10]:


# compute the number of NaN values for each column
drop_nan = data_copy.isnull().sum()

# get the column having NaN value more than 30%
drop_nan = drop_nan[drop_nan.values > (len(data_copy) * 0.30)]


# In[11]:


# get the column name
drop_nan.index


# In[12]:


len(drop_nan)


# In[13]:


# delete those columns having Missing values more than 30%, because it is not wise to keep column 
# having most of the values are missing
data_copy.drop(labels= drop_nan.index, inplace = True, axis = 1)


# In[14]:


data_copy.head(5)


# In[15]:


# compute the correlation matrix to know the name of all the columns which are dependent on each other
# if two columns (Also know as features or attributes) are dependent it means keeping one if enough for prediction
# we can reproduce the another feature from 1st one. hence delete the dependent feature
# even if we keep the dependent feature, it will not contribute in improving the accuracy.
# but it will make the process slow because of the unnecessary features.

corr_matrix  = data_copy.corr().abs()


# # remove dependent (highly correlated) features (correlation coeff > 0.98)

# In[16]:


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation value greater than 0.98 (i.e. features to be strongly dependent)
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

# Drop features 
data_copy.drop(to_drop, axis=1, inplace=True)


# In[17]:


# this is the list of features which we need to remove because of the highly dependency.
to_drop


# In[18]:


# based on preliminary observation, these are the extra useless columns which will not help in prediction. 
col_drop = ['id', 'emp_title', 'issue_d', 'pymnt_plan', 'url', 'title', 'zip_code', 'addr_state',
            'earliest_cr_line', 'initial_list_status', 'out_prncp', 'total_pymnt', 'last_pymnt_d', 'last_pymnt_amnt',
            'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'policy_code', 'hardship_flag',
            'disbursement_method', 'debt_settlement_flag']

# drop these features as well
data_copy.drop(columns =col_drop, inplace = True)


# In[19]:


# check the shape of the new df
data_copy.shape


# In[20]:


# get the loan status and their respective count
data_copy['loan_status'].value_counts()


# In[21]:


# replace 'Does not meet the credit policy. Status:Charged Off' as charged off
# and 'Does not meet the credit policy. Status:Fully Paid' as Fully Paid

data_copy['loan_status'].replace(['Does not meet the credit policy. Status:Fully Paid','Does not meet the credit policy. Status:Charged Off'],['Fully Paid','Charged Off'],inplace=True)


# In[22]:


data_copy['loan_status'].value_counts()


# In[23]:


# Now we only consider Fully paid and charged off only
data_copy = data_copy[(data_copy['loan_status']=='Fully Paid') | (data_copy['loan_status']=='Charged Off')]


# In[24]:


data_copy.to_csv('data/processed_data.csv')


# In[25]:


data_copy['loan_status'].value_counts()


# In[26]:


# visualize on the bar plot the count of 'fully paid' and 'Default'
plt.ticklabel_format(style='plain')
t = pd.value_counts(data_copy['loan_status'].values, sort=True)
t.plot.barh(color=['g','r'])
print(data_copy['loan_status'].value_counts(normalize=True)*100)
plt.show()


# In[27]:


# check for NaN values again
data_copy.isnull().sum()


# ## Deal with NAN

# In[28]:


for col in data_copy.columns:
    
    # replace float attributes with their median value
    if isinstance(data_copy[col][0], float):
        data_copy[col].fillna(data_copy[col].median(), inplace = True)
        
        # replace other attribute with their mode value
    else:
        data_copy[col].fillna(data_copy[col].mode()[0], inplace = True)


# In[29]:


# check NAN again
data_copy.isnull().sum()


# - Now there is no missing values.

# In[30]:


data_copy.dtypes


# # categorical features

# In[31]:


# get the categorical featues
# The columns with object dtype are the possible categorical features in a dataset.
category_column = data_copy.dtypes.index[data_copy.dtypes=='object']


# In[32]:


# print categorical features and the count of their unique values.
# loop to print categorical features that are stored in the category_column variable
for i in category_column:
    print(i)
    print(data_copy[i].value_counts())
    print(20*'-')


# In[33]:


# there are lot of classification for sub_grade, hence delete it. Also its sub feature of grade
data_copy.drop('sub_grade', axis=1, inplace=True)


# In[34]:


# Convert categorical features to neumerical values
data_copy['term'].replace((' 36 months', ' 60 months'),(36 ,60), inplace = True) 
data_copy['grade'].replace(('A','B','C','D','E','F','G'),(1,2,3,4,5,6,7), inplace = True) 
data_copy['emp_length'].replace(('10+ years','2 years','< 1 year','3 years','1 year','5 years','4 years','6 years','8 years','7 years','9 years'),(10,2,0.5,3,1,5,4,6,8,7,9), inplace  = True)
data_copy['home_ownership'].replace(('MORTGAGE', 'RENT','OWN','ANY', 'OTHER','NONE'),(1,2,3,4,5,6), inplace = True) 
data_copy['verification_status'].replace(('Source Verified', 'Verified','Not Verified'),(1,2,3), inplace = True) 
data_copy['loan_status'].replace(('Fully Paid', 'Charged Off'),(0,1), inplace = True) 
data_copy['purpose'].replace(('debt_consolidation', 'credit_card','home_improvement','other', 'major_purchase','medical','small_business','car','moving','vacation','house','wedding','renewable_energy','educational'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14), inplace = True) 
data_copy['application_type'].replace(('Individual','Joint App'),(1,2), inplace  = True)


# # Imbalance data

# In[35]:


# percentage of paid /unpaid
data_copy['loan_status'].value_counts(normalize=True)*100


# In[36]:


# training and test set splitting

from sklearn.model_selection import train_test_split

# get x and y
x = data_copy.drop(columns='loan_status',axis=1)
y = data_copy['loan_status']

# feature scaling to bring the features into same range
# StandardScaler() transforms the data in such a manner that it has mean as 0 and standard deviation as 1.

scaler = StandardScaler()
scaler_data = scaler.fit_transform(x)

# split the data. 70% for training and 30% for testing
# this is a pretty common split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, shuffle = True, random_state=42)


# # Baseline Models

# In[37]:


from sklearn.dummy import DummyClassifier
# DummyClassifier to predict only target 0
dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
dummy_pred = dummy.predict(x_test)

# checking unique labels
print('Unique predicted labels: ', (np.unique(dummy_pred)))

# checking accuracy
print('Test score: ', accuracy_score(y_test, dummy_pred))


# In[38]:


roc_auc_score(y_test, dummy_pred)


# - 0 - Fully Paid
# - 1 - Charged Off
# 
# As predicted our accuracy score for classifying all Loan as Fully Paid is 80.079%!
# 
# As the Dummy Classifier predicts only Class 1 (i.e. Fully Paid), it is clearly not a good option for our objective of correctly classifying.
# 
# Let's see how logistic regression performs on this dataset.

# # Logistic Regression

# In[87]:


# build the model
lr_model = LogisticRegression(random_state=42,solver='saga')

# fit the model on training data
lr_model.fit(x_train,y_train)

# make prediction on test data
y_pred = lr_model.predict(x_test)


# In[88]:


# get the accuracy
accuracy_score(y_test, y_pred)


# In[89]:


# Checking unique values
predictions = pd.DataFrame(y_pred)
predictions[0].value_counts()


# Logistic Regression outperformed the Dummy Classifier! We can see that it predicted 61197 instances of class 1 (i.e. charged off), so this is definitely an improvement. But can we do better?
# 
# Let's see if we can apply some techniques for dealing with class imbalance to improve these results.
# 
# Accuracy is not the best metric to use when evaluating imbalanced datasets as it can be misleading. Metrics that can provide better insight include:
# 
# - Confusion Matrix: a table showing correct predictions and types of incorrect predictions.
# - Precision: the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier's exactness. Low precision indicates a high number of false positives.
# - Recall: the number of true positives divided by the number of positive values in the test data. Recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier's completeness. Low recall indicates a high number of false negatives.
# - F1: Score: the weighted average of precision and recall.
# - Since our main objective with the dataset is to prioritize accuraltely classifying loan status, the recall score can be considered our main metric to use for evaluating outcomes.

# In[90]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
print('recall: ' + str(recall_score(y_test, y_pred)))
print('precision: ' + str(precision_score(y_test, y_pred)))
print('auc: ' + str(roc_auc_score(y_test, y_pred)))


# In[91]:


pd.DataFrame(confusion_matrix(y_test, y_pred),
             columns=['true_fully_paid','true_charged_off'],
             index=['predict_fully_paid','predict_charged_off'])


# In[92]:


plot_confusion_matrix(lr_model, x_test, y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=['Fully-Paid','Charged Off'])


# We have a very high accuracy score of 0.99 And from the confusion matrix, we can see we are misclassifying several observations leading to a recall score of 0.986 only. We will try other method like oversampling/undersampling further.

# # Up-sampling

# In[45]:


X = pd.concat([x_train, y_train], axis=1)
fully_paid = X[X['loan_status']==0]
charged_off =  X[X['loan_status']==1]
from sklearn.utils import resample
# upsample minority
charged_off_upsampled = resample(charged_off,
                          replace=True, # sample with replacement
                          n_samples=len(fully_paid), # match number in majority class
                          random_state=42) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([fully_paid, charged_off_upsampled])

# check new class counts
upsampled['loan_status'].value_counts()


# In[46]:


# trying logistic regression again with the balanced dataset
y_train_upsampled = upsampled['loan_status']
X_train_upsampled = upsampled.drop('loan_status', axis=1)

lr_model_upsampled = LogisticRegression(random_state=42,solver='saga')
lr_model_upsampled.fit(X_train_upsampled, y_train_upsampled)

upsampled_pred = lr_model_upsampled.predict(x_test)

# Checking accuracy
print('accuracy: ' + str(accuracy_score(y_test, upsampled_pred)))
print('recall: ' + str(recall_score(y_test, upsampled_pred)))
print('precision: ' + str(precision_score(y_test, upsampled_pred)))
print('auc: ' + str(roc_auc_score(y_test, upsampled_pred)))


# # 2 .Undersampling Majority Class
# Undersampling can be defined as removing some observations of the majority class. Undersampling can be a good choice when you have a ton of data -think millions of rows. But a drawback to undersampling is that we are removing information that may be valuable. Undersampling can be defined as removing some observations of the majority class. This is done until the majority and minority class is balanced out

# In[47]:


# still using our separated classes fraud and not_fraud from above

# downsample majority
fully_paid_downsampled = resample(fully_paid,
                                replace = False, # sample without replacement
                                n_samples = len(charged_off), # match minority n
                                random_state = 42) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([fully_paid_downsampled, charged_off])

# checking counts
downsampled['loan_status'].value_counts()


# In[48]:


# trying logistic regression again with the balanced dataset
y_train_downsampled = downsampled['loan_status']
X_train_downsampled = downsampled.drop('loan_status', axis=1)

downsampled_lr = LogisticRegression(random_state= 42,solver='saga').fit(X_train_downsampled, y_train_downsampled)

downsampled_pred = downsampled_lr.predict(x_test)
# Checking accuracy
print('accuracy: ' + str(accuracy_score(y_test, downsampled_pred)))
print('recall: ' + str(recall_score(y_test, downsampled_pred)))
print('precision: ' + str(precision_score(y_test, downsampled_pred)))
print('auc: ' + str(roc_auc_score(y_test, downsampled_pred)))


# In[49]:


# confusion matrix for without sampling,upsampling and downsampling

f, axs = plt.subplots(nrows=3, ncols=1,figsize=(5,10))

plot_confusion_matrix(lr_model, x_test, y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=['Fully-Paid','Charged Off'],ax=axs[0])

plot_confusion_matrix(lr_model_upsampled, x_test, y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=['Fully-Paid','Charged Off'],ax=axs[1])

plot_confusion_matrix(downsampled_lr, x_test, y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=['Fully-Paid','Charged Off'],ax=axs[2])

# axs[0].grid(False)
axs[0].set_xlabel('Without Sampling')
axs[1].set_xlabel('Up sampled')
axs[2].set_xlabel('Down Sampled')
f.tight_layout()
f.savefig('cm.png') 


# # Apply Grid Search on downsampled data with cross validation and regularizion
# - to improve the accuracy
# 
# Downsampling is a mechanism that reduces the count of training samples falling under the majority class. Even out counts of target categories to handle imbalanced data
# 
# Grid search = the holy grail: it is a technique which will pass different values of C and Penalty variable to the model and check at which combination of these values model is performing best
# And accordingly return those values of C and Penalty at which model perfomed best

# In[106]:


from sklearn.model_selection import GridSearchCV
LR = LogisticRegression()

# Hyperparamaters
# A model is tried for each l1 and 0.001, l1 ad 0.01, l1 and 0.1, etc.
# A model is also tried for l2 and 0.001, l2 and 0.01, and l2 0.01 etc.

# penalty is for regularization, and we penalize many independent variables to prevent overfitting
# we want to generalize a model, not fit to data perfectly because future data may not have unique outliers
# we generalize to moel

# C is regularization parameter
# C is just lowering it to increase the regularization strength, it is a hyperparameter (set before training)
# C = 1/lambda so the lower the C, the more extreme the regularization
# In sci-kit 'C' parameter is inverse of lambda. Sci kit does implemented 'C' parameter to standardise


# the penalty is regularization
# C controls the strength of it (regularization) and 
# it is inverse regularization

LRparam_grid  = {'solver' : ['saga'],
      'penalty' : ['elasticnet', 'l1', 'l2', 'none'],
      'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 


# grid search with 3 fold cross validation, since dataset is huge , even 3 fold is enough
logreg_cv = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 1, cv = 3, scoring='roc_auc',n_jobs=-1)
logreg_cv.fit(X_train_downsampled, y_train_downsampled)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# In[124]:


c = logreg_cv.best_params_['C']
p = logreg_cv.best_params_['penalty']
s = logreg_cv.best_params_['solver']

log_model_downsampled = LogisticRegression(C=c, penalty=p, random_state= 42, solver = s)
log_model_downsampled.fit(X_train_downsampled, y_train_downsampled)
y_pred_down = log_model_downsampled.predict(x_test)
roc_auc_score(y_test, y_pred_down)


# In[129]:


print('begin')
log_model_downsampled = LogisticRegression(C=0.01, penalty='l2', random_state= 42, solver = 'newton-cg')
log_model_downsampled.fit(X_train_downsampled, y_train_downsampled)
y_pred_down = log_model_downsampled.predict(x_test)
roc_auc_score(y_test, y_pred_down)


# In[130]:


accuracy_score(y_test, y_pred_down)


# In[ ]:





# # Apply Grid Search on upsampled data with cross validation and regularizion
# 
# #Upsampling is a procedure where synthetically generated data points (corresponding to minority class) are injected into the dataset. After this process, the counts of both labels are almost the same. This equalization procedure prevents the model from inclining towards the majority class

# In[110]:


from sklearn.model_selection import GridSearchCV
LR = LogisticRegression()

# paramaters

# LRparam_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100,1000],
#     'penalty': ['l1', 'l2'],
# } 

LRparam_grid  = {'solver' : ['saga'],
      'penalty' : ['elasticnet', 'l1', 'l2', 'none'],
      'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



# grid search with 3 fold cross validation, since dataset is huge , even 3 fold is enough
logreg_cv = GridSearchCV(LR, param_grid=LRparam_grid, refit = True, verbose = 1, cv=3,scoring='roc_auc',n_jobs=-1)
logreg_cv.fit(X_train_upsampled, y_train_upsampled)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# In[111]:


c = logreg_cv.best_params_['C']
p = logreg_cv.best_params_['penalty']
s = logreg_cv.best_params_['solver']

log_model_upsampled = LogisticRegression(C=c, penalty=p, random_state= 42, solver = s)
log_model_upsampled.fit(X_train_upsampled, y_train_upsampled)
y_pred_up = log_model_upsampled.predict(x_test)
roc_auc_score(y_test, y_pred_up)


# In[131]:


print('begin')
log_model_upsampled = LogisticRegression(C=0.01, penalty='l2', random_state= 42, solver = 'newton-cg')
log_model_upsampled.fit(X_train_upsampled, y_train_upsampled)
y_pred_up = log_model_upsampled.predict(x_test)
roc_auc_score(y_test, y_pred_up)


# In[132]:


accuracy_score(y_test, y_pred_up)


# # Results

# In[113]:


# Support is a clarification report which displays all types of classification results
# Just understand recall, precision, F1, and auc_roc_score
# Don't worry about Weighted avg, micro etc.

print('***********Default Logistic Regression*************roc_auc_score:',roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('*********** Logistic Regression- upsampled*************roc_auc_score:',roc_auc_score(y_test, upsampled_pred))
print(classification_report(y_test, upsampled_pred))

print('*********** Logistic Regression- downsampled*************roc_auc_score:',roc_auc_score(y_test, downsampled_pred))
print(classification_report(y_test, downsampled_pred))

print('*********** Logistic Regression-upsampled- Grid search*************roc_auc_score:',roc_auc_score(y_test, y_pred_up))
print(classification_report(y_test, y_pred_up))

print('*********** Logistic Regression-downsampled-Grid search*************roc_auc_score:',roc_auc_score(y_test, y_pred_down))
print(classification_report(y_test, y_pred_down))


# In[ ]:





# In[117]:


from xgboost import XGBClassifier
from skopt import BayesSearchCV

xgbc=XGBClassifier(scale_pos_weight=10)
param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,6,7,8,9,10,11,12,13,14],
              'n_estimators': [50,65,80,100,115,130,150],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
clf = BayesSearchCV(estimator=xgbc, search_spaces=param_grid,cv=3, return_train_score=True, verbose=3)
clf.fit(x_train, y_train)


# In[136]:


import matplotlib.pyplot as plt
import scienceplots

vi=pd.DataFrame(log_model_upsampled.coef_[0]*np.std(x, 0),columns=['vi'])
vi=vi.sort_values(by='vi')
vi['true_im']=abs(vi['vi'])

vi_=vi.iloc[:10,:]
vi_['true_im'][0]=(1/10)*vi_['true_im'][0]
with plt.style.context(['science','ieee','no-latex']):
    plt.barh(range(len(vi_['true_im'])),vi_['true_im'],tick_label = vi_.index)
    plt.savefig('vi_lr.png')


# In[137]:


# upsampled
f, axs = plt.subplots(figsize=(7,7))
plot_roc_curve(log_model_upsampled, x_test, y_test,ax=axs)
plt.savefig('AUC.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[50]:


sns.heatmap(data_copy.corr());


# In[51]:


data_copy.shape


# In[ ]:





# In[ ]:





# In[52]:


vi=pd.DataFrame(lr_model.coef_[0]*np.std(x, 0),columns=['vi'])
vi=vi.sort_values(by='vi')
vi['true_im']=abs(vi['vi'])

