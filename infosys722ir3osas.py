
# coding: utf-8

# # Import the data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df1 = pd.read_csv('./Datasets/adult.data', skipinitialspace=True)
df2 = pd.read_csv('./Datasets/adult.test', skipinitialspace=True)

df = df1.append(df2, ignore_index=True)


# ### Explore the data

# In[3]:


df


# In[4]:


df.dtypes


# In[5]:


pd.value_counts(df['income-classify'])


# In[6]:


df.replace(["<=50K.", ">50K."], ["<=50K", ">50K"], inplace=True)


# In[7]:


pd.value_counts(df['income-classify'])


# In[8]:


df.describe().round(2)


# In[9]:


df.groupby('income-classify').mean().round(2).reset_index().rename(columns={'age':'Avg age',                                                                           'fnlwgt':'Avg fnlwgt',                                                                           'education-num':'Avg education-num',                                                                           'capital-gain':'Avg capital-gain',                                                                           'capital-loss':'Avg capital-loss',                                                                           'hours-per-week':'Avg hours-per-week'})


# In[10]:


df.groupby([df['education-num'],df['education']])['age','hours-per-week'].mean().round(2).rename(columns={'age':'Avg age',                'hours-per-week':'Avg hours-per-week'})


# ### Data visualization

# In[11]:


# plot the distribution of education, sex, workclass, income-classify, 
#occupation, race, marital-status and relationship

plt.figure()
df.groupby(['education']).size().plot(kind='bar')

plt.figure()
df.groupby(['sex']).size().plot(kind='bar')

plt.figure()
df.groupby(['workclass']).size().plot(kind='bar')

plt.figure()
df.groupby(['income-classify']).size().plot(kind='bar')

plt.figure()
df.groupby(['occupation']).size().plot(kind='bar')

plt.figure()
df.groupby(['race']).size().plot(kind='bar')

plt.figure()
df.groupby(['marital-status']).size().plot(kind='bar')

plt.figure()
df.groupby(['relationship']).size().plot(kind='bar')


# In[12]:


# age vs income
df.groupby(['age', 'income-classify']).size().unstack().plot(kind='bar', figsize=(15, 7.5))
plt.title('Relationship between age and income')


# In[13]:


plt.figure()
df.groupby(['hours-per-week', 'income-classify']).size().unstack().plot(kind='bar', figsize=(15, 7.5), stacked = True)
plt.title('Relationship between working hours and income')


# In[14]:


# education vs income
df.groupby(['education-num', 'income-classify']).size().unstack().plot(kind='bar', figsize=(15, 7.5),stacked=True)
plt.title('Relationship between education and income')


# In[15]:


# sex vs income
df.groupby(['income-classify', 'sex']).size().plot(kind='pie', figsize=(7.5, 7.5))
plt.title('Relationship between gender and income')


# In[16]:


# education num vs avg hours per week
df.groupby(['education-num'])['hours-per-week'].mean().plot(kind='line', figsize=(15, 7.5))
plt.title('Relationship between education and working hours')


# In[17]:


# age vs avg hours per week
df.groupby(['age'])['hours-per-week'].mean().plot(kind='line', figsize=(15, 7.5))
plt.title('Relationship between average working hours')


# In[18]:


# race vs avg hours per week
df.groupby(['race'])['hours-per-week'].mean().plot(kind='bar', figsize=(15, 7.5))
plt.title('Relationship between race and average working hours')


# In[19]:


# race vs income
df.groupby(['race', 'income-classify']).size().unstack().plot(kind='bar', figsize=(15, 7.5))
plt.title('Relationship between race and income')


# In[20]:


df.groupby(['marital-status', 'income-classify']).size().unstack().plot(kind='pie', figsize=(15, 7.5), subplots=True)


# In[21]:


df.groupby(['relationship', 'income-classify']).size().unstack().plot(kind='pie', figsize=(15, 7.5), subplots=True)


# ##### Temporary variable used to identify how many missing values

# In[22]:


# MisValTemp = df.replace('?', np.NaN)


# In[23]:


# MisValTemp


# In[24]:


# MisValTemp.dropna(inplace=True)
# len(MisValTemp)


# ##### Temporary variable used to identify how many extreme values

# In[25]:


# ExtrValTemp = df[(df['age'] < 74)]


# In[26]:


# len(ExtrValTemp)


# In[27]:


# ExtrValTemp = df[(df['hours-per-week'] >= 8.5) & (df['hours-per-week'] <= 72.4)]


# In[28]:


# len(ExtrValTemp)


# ##### Temporary variable used to identify how many extreme values

# In[29]:


# DupValTemp = df.drop_duplicates()


# In[30]:


# len(DupValTemp)


# #### Drop irrelevant features

# In[31]:


df = df.drop(['fnlwgt', 'education'], axis=1)
print("Number of columns: " + str(df.columns.size))


# #### Dealing with missing, extreme and duplicated values

# In[32]:


df = df[(df['age'] < 74) & (df['hours-per-week'] >= 8.5) & (df['hours-per-week'] <= 72.4)]
df = df.replace('?', np.nan).dropna()

df = df.drop_duplicates()

len(df)


# ### Data projection

# #### Reclassify education 

# In[33]:


edu_unm_level = {
    1:'primary',
    2:'primary',
    3:'primary',
    4:'primary',
    5:'secondary',
    6:'secondary',
    7:'secondary',
    8:'secondary',
    9:'secondary',
    10:'undergraduate',
    11:'undergraduate',
    12:'undergraduate',
    13:'undergraduate',
    14:'higher_than_Master',
    15:'higher_than_Master',
    16:'higher_than_Master'
}


# In[34]:


df['edu_level'] = df['education-num'].map(edu_unm_level)


# #### Reclassify marital status

# In[35]:


marital_status = {
    'Married-civ-spouse':'Married',
    'Married-spouse-absent':'Married',
    'Married-AF-spouse':'Married',
    'Divorced':'Divorced',
    'Never-married':'Never_married',
    'Separated':'Separated',
    'Widowed':'Widowed',
}


# In[36]:


df['marital_status'] = df['marital-status'].map(marital_status)


# #### Reclassify countires

# In[37]:


df['native-country'].unique()


# In[38]:


countries = {
    'Cambodia':'SE-Asia',
    'Canada':'British-Commonwealth',
    'China':'China',
    'Columbia':'South-America',
    'Cuba':'South-America',
    'Dominican-Republic':'South-America',
    'Ecuador':'South-America',
    'El-Salvador':'South-America',
    'England':'British-Commonwealth',
    'France':'Euro',
    'Germany':'Euro',
    'Greece':'Euro',
    'Guatemala':'South-America',
    'Haiti':'South-America',
    'Holand-Netherlands':'Euro',
    'Honduras':'South-America',
    'Hong':'China',
    'Hungary':'Euro',
    'India':'British-Commonwealth',
    'Iran':'Euro',
    'Ireland':'British-Commonwealth',
    'Italy':'Euro',
    'Jamaica':'South-America',
    'Japan':'APAC',
    'Laos':'SE-Asia',
    'Mexico':'South-America',
    'Nicaragua':'South-America',
    'Outlying-US(Guam-USVI-etc)':'South-America',
    'Peru':'South-America',
    'Philippines':'SE-Asia',
    'Poland':'Euro',
    'Portugal':'Euro',
    'Puerto-Rico':'South-America',
    'Scotland':'British-Commonwealth',
    'South':'Euro',
    'Taiwan':'China',
    'Thailand':'SE-Asia',
    'Trinadad&Tobago':'South-America',
    'United-States':'United-States',
    'Vietnam':'SE-Asia',
    'Yugoslavia':'Euro'
}


# In[39]:


df['native_country'] = df['native-country'].map(countries)


# In[40]:


pd.value_counts(df['native_country'])


# #### One hot encoding

# In[41]:


# df['workclass'].replace('Private', 1, inplace=True)


# In[42]:


# df['workclass'].unique()


# In[43]:


# df['workclass'].replace('State-gov', '1', inplace=True)
# df['workclass'].replace('Self-emp-not-inc', '2', inplace=True)
# df['workclass'].replace('Private', '3', inplace=True)
# df['workclass'].replace('Federal-gov', '4', inplace=True)
# df['workclass'].replace('Local-gov', '5', inplace=True)
# df['workclass'].replace('Self-emp-inc', '6', inplace=True)
# df['workclass'].replace('Without-pay', '7', inplace=True)

# df['marital_status'].replace('Married', '1', inplace=True)
# df['marital_status'].replace('Never_married', '2', inplace=True)
# df['marital_status'].replace('Divorced', '3', inplace=True)
# df['marital_status'].replace('Separated', '4', inplace=True)
# df['marital_status'].replace('Widowed', '5', inplace=True)

# df['occupation'].replace('Exec-managerial', '1', inplace=True)
# df['occupation'].replace('Protective-serv', '2', inplace=True)
# df['occupation'].replace('Craft-repair', '3', inplace=True)
# df['occupation'].replace('Other-service', '4', inplace=True)
# df['occupation'].replace('Farming-fishing', '5', inplace=True)
# df['occupation'].replace('Prof-specialty', '6', inplace=True)
# df['occupation'].replace('Sales', '7', inplace=True)
# df['occupation'].replace('Tech-support', '8', inplace=True)
# df['occupation'].replace('Adm-clerical', '9', inplace=True)
# df['occupation'].replace('Transport-moving', '10', inplace=True)
# df['occupation'].replace('Handlers-cleaners', '11', inplace=True)
# df['occupation'].replace('Machine-op-inspct', '12', inplace=True)
# df['occupation'].replace('Priv-house-serv', '13', inplace=True)
# df['occupation'].replace('Armed-Forces', '14', inplace=True)

# df['relationship'].replace('Not-in-family', '1', inplace=True)
# df['relationship'].replace('Husband', '2', inplace=True)
# df['relationship'].replace('Wife', '3', inplace=True)
# df['relationship'].replace('Unmarried', '4', inplace=True)
# df['relationship'].replace('Other-relative', '5', inplace=True)
# df['relationship'].replace('Own-child', '6', inplace=True)

# df['race'].replace('White', '1', inplace=True)
# df['race'].replace('Asian-Pac-Islander', '2', inplace=True)
# df['race'].replace('Black', '3', inplace=True)
# df['race'].replace('Amer-Indian-Eskimo', '4', inplace=True)
# df['race'].replace('Other', '5', inplace=True)

# df['sex'].replace('Female', '1', inplace=True)
# df['sex'].replace('Male', '2', inplace=True)

df['income-classify'].replace('<=50K', '0', inplace=True)
df['income-classify'].replace('>50K', '1', inplace=True)

# df['edu_level'].replace('primary', '1', inplace=True)
# df['edu_level'].replace('secondary', '2', inplace=True)
# df['edu_level'].replace('undergraduate', '3', inplace=True)
# df['edu_level'].replace('higher_than_Master', '4', inplace=True)

# df['native_country'].replace('United-States', '1', inplace=True)
# df['native_country'].replace('South-America', '2', inplace=True)
# df['native_country'].replace('British-Commonwealth', '3', inplace=True)
# df['native_country'].replace('Euro', '4', inplace=True)
# df['native_country'].replace('SE-Asia', '5', inplace=True)
# df['native_country'].replace('China', '6', inplace=True)
# df['native_country'].replace('APAC', '7', inplace=True)


# In[44]:


df = df.drop(['education-num', 'marital-status', 'native-country'], axis=1)


# In[45]:


# # One Hot Encodes all labels before Machine Learning
one_hot_cols = df.drop(['income-classify','age', 'capital-gain', 'capital-loss', 'hours-per-week'], axis=1).columns.tolist()
df_enc = pd.get_dummies(df, columns=one_hot_cols)
df_enc.head(10)


# ### Rebalancing the data

# In[46]:


labels = []
for i, dfi in enumerate(df_enc.groupby(['income-classify'])):
    labels.append(dfi[0])
    plt.bar(i, dfi[1].count(), label=dfi[0])
plt.xticks(range(len(labels)), labels)
plt.legend()
plt.show()


# In[47]:


target_count = df_enc['income-classify'].value_counts()
print('Class 1:', target_count[0], ', % =', target_count[0]/target_count.sum())
print('Class 2:', target_count[1], ', % =', target_count[1]/target_count.sum())
print(target_count.sum())


# In[48]:


count_class_1, count_class_2 = df['income-classify'].value_counts()

# Divide by class
df_class_1 = df_enc[df_enc['income-classify'] == '0']
df_class_2 = df_enc[df_enc['income-classify'] == '1']

# df_class_1 = df[df['income-classify'] == '<=50K']
# df_class_2 = df[df['income-classify'] == '>50K']

df_class_2_under = df_class_2.sample(count_class_1,replace=True)
df_enc = pd.concat([df_class_2_under,df_class_1], axis=0)

print('Random Under-Sampling:')
print(df_enc['income-classify'].value_counts())

df_enc['income-classify'].value_counts().plot(kind='bar', title='Count (income-classify)');


# ## Data mining

# ### Split the data

# In[49]:


train = df_enc.sample(frac=0.8, random_state=1)
test = df_enc.loc[~df.index.isin(train.index)]


# In[50]:


train.shape


# In[51]:


test.shape


# In[52]:


train['income-classify'].value_counts()


# In[53]:


x_train = train.drop(['income-classify'], axis=1)
y_train = train['income-classify']

x_test = test.drop(['income-classify'], axis=1)
y_test = test['income-classify']


# #### Define the roc plot function

# In[54]:


from sklearn import metrics
def plot_roc_curve(y_test, preds):
    y_true=list(map(int, y_test))
    y_pred=list(map(int, preds))
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# ### Function for plot confusion matrix

# In[55]:


import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ### Decision Tree

# In[56]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

dt_model = tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=40,                                      min_samples_split=0.02,                                      min_samples_leaf=0.01).fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

# model accuracy for X_test  
print("============================================================")
accuracy_dt = dt_model.score(x_test, y_test)
print('Decision Forest classifier accuracy: ', accuracy_dt)

y_true = list(y_test)
y_pred = list(dt_pred)
print("============================================================")
print("")
print(metrics.classification_report(y_test, y_pred))
print("============================================================")
print("")

plot_roc_curve(y_true, y_pred)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['<=50K', '>50K'],
                      title='Confusion matrix')


# In[57]:


# dt_model.get_params()


# #### Visualize the tree. Note graphviz library required to run this part

# In[58]:


# import graphviz

# cols = x_train.columns
# target_names = ['less50K', 'grt50K']
# dot_data = tree.export_graphviz(dt_model, out_file=None, label='all', feature_names=cols,\
#                                 class_names=target_names, filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render('tree_view')


# ### SVM

# In[59]:


from sklearn import svm
svm_model = svm.SVC(C=0.8).fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)

# # creating a confusion matrix
print("============================================================")
accuracy_svm = svm_model.score(x_test, y_test)
print('SVM classifier accuracy: ', accuracy_svm)

y_true = list(y_test)
y_pred = list(svm_pred)
print("============================================================")
print("")
print(metrics.classification_report(y_test, y_pred))
print("============================================================")
print("")
print(confusion_matrix(y_true, y_pred))
print("")
print("============================================================")

plot_roc_curve(y_true, y_pred)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['<=50K', '>50K'],
                      title='Confusion matrix')


# In[60]:


# svm_model.get_params()


# ### Logistic regression

# In[61]:


from sklearn import linear_model
from sklearn.metrics import confusion_matrix

lr_model = linear_model.LogisticRegression(class_weight='balanced', solver='lbfgs',                                           multi_class='multinomial', C=0.8).fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

# model accuracy for X_test  
print("============================================================")
accuracy_lr = lr_model.score(x_test, y_test)
print('Logistic Regression classifier accuracy: ', accuracy_lr)

y_true = list(y_test)
y_pred = list(lr_pred)
print("============================================================")
print("")
print(metrics.classification_report(y_test, y_pred))
print("============================================================")
print("")
print(confusion_matrix(y_true, y_pred))
print("")
print("============================================================")

plot_roc_curve(y_true, y_pred)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['<=50K', '>50K'],
                      title='Confusion matrix')


# In[62]:


# lr_model.get_params()


# ### Nerual network

# In[63]:


from sklearn import neural_network
from sklearn.metrics import confusion_matrix

nn_model = neural_network.MLPClassifier(hidden_layer_sizes=(150, ), activation='relu',                                        solver='adam', learning_rate='adaptive').fit(x_train, y_train)
nn_pred = nn_model.predict(x_test)

# # model accuracy for X_test  
print("============================================================")
accuracy_nn = nn_model.score(x_test, y_test)
print('Logistic Regression classifier accuracy: ', accuracy_nn)

y_true = list(y_test)
y_pred = list(nn_pred)
print("============================================================")
print("")
print(metrics.classification_report(y_test, y_pred))
print("============================================================")
print("")
print(confusion_matrix(y_true, y_pred))
print("")
print("============================================================")

plot_roc_curve(y_true, y_pred)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['<=50K', '>50K'],
                      title='Confusion matrix')


# In[64]:


# nn_model.get_params()


# In[ ]:




