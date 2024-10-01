#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.tree as tr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[8]:


# import dataset
data=pd.read_csv('C:/Users/hp/Downloads/Breast_Cancer.csv')

print(data)
data.head()

x = data.drop("Status", axis=1)
y = data["Status"]


# In[9]:


# Total no of Alive & Dead
data['Status'].value_counts()


# In[10]:


# Age distribution by Status
plt.figure(figsize=(10, 6))

sns.histplot(x='Age', hue='Status', kde=True, element='step', common_norm=False, data=data)
plt.title('Age Distribution by Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Status', labels=['Alive', 'Dead'])
plt.show()


# In[11]:


print(data.columns)


# In[12]:


# count plot relationship between Race and Status
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Race', hue='Status', data=data)

# add annotations to display total counts on each bar
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center', xytext=(0, 10),textcoords='offset points')


plt.title('Relationship between Race and Status')
plt.xlabel('Race')
plt.ylabel('Count')
plt.legend(title='Status')
plt.show()


# In[13]:


data.shape
data.info()
# Removing unwanted columns
data = data.drop(["Marital Status", "differentiate"], axis=1) 


# In[ ]:





# In[14]:


data.describe()


# In[15]:


# Label Encoding to binary columns

data["Survival Months"] = np.where(data["Survival Months"] > data["Survival Months"].max() // 2, 1, 0) # Translate column into binary values; makes it easier
data["Status"] = np.where(data["Status"] == "Alive", 1, 0)
data["Estrogen Status"] = np.where(data["Estrogen Status"] == "Positive", 1, 0) # Positive = 1, Negative = 0
data["Progesterone Status"] = np.where(data["Progesterone Status"] == "Positive", 1, 0)
data = pd.get_dummies(data) # One-hot encoding, changes categorical data into numerical data
data.head()


# In[ ]:





# In[16]:


from sklearn import tree
x = data.drop("Status", axis=1)
y = data["Status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



DT=tr.DecisionTreeClassifier(max_depth=3)
DT.fit(x_train,y_train)
trACC_DT=DT.score(x_train,y_train)
tesACC_DT=DT.score(x_test,y_test)

print("train accuracy is: ",trACC_DT)
print("Test accuracvy is :",tesACC_DT)
model = DT.fit(x_train, y_train)
text_representation = tr.export_text(DT)
print(text_representation)

# To Visualize Decision Tree
tree.plot_tree(model)


# In[17]:


# Decision tree
trACC=[]
tesACC=[]
MD=[]

for i in range(2,8):
    #Create a Decision Tree classifier with the current max_depth
    DT=tr.DecisionTreeClassifier(max_depth=i)
    DT.fit(x_train,y_train)
    trACC.append(DT.score(x_train,y_train))
    tesACC.append(DT.score(x_test,y_test))
    MD.append(i)
#print(trACC)
#print(tesACC)
#print(MD)
plt.figure()
plt.plot(MD, trACC, label='Train',marker='o')
plt.plot(MD, tesACC, label='Test', marker='o')
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#print(trACC)
#print(tesACC)


# In[18]:


#Evaluating of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
 
#Calculate accuracy

# Make predictions on the test set
y_pred = DT.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Prediction Accuracy for Decesion Tree :", accuracy)

# Calculate F1 score
f1=f1_score(y_test, y_pred)
print("F1 Score(DT):", f1)
print("Length of y_tes:", len(y_test))
print("Length of y_pred:", len(y_pred))
    
    
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision(DT):", precision)
print("Recall:", recall)
 
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (DT):")
print(conf_matrix)


# In[61]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
# 2. Make predictions on your test data
y_pred_proba = DT.predict_proba(x_test)[:, 1]

# 3. Calculate false positive rate (FPR) and true positive rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# 4. Calculate area under the ROC curve (AUC)
auc = roc_auc_score(y_test, y_pred_proba)

# 5. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[13]:


#KNN


# In[62]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.tree as tr
import sklearn.neighbors as ne 
import sklearn.naive_bayes as nb
import sklearn.linear_model as lm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# In[110]:


#KNN=ne.KNeighborsClassifier(n_neighbors=5)
KNN = ne.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
KNN.fit(x_train,y_train)
KNN.fit(x_test,y_test)


# In[111]:


y_pred = KNN.predict(x_test.values)
print(y_pred)


# In[112]:


trACC_KNN = KNN.score(x_train.values, y_train)
tesACC_KNN = KNN.score(x_test.values, y_test)

print('Train Accuracy for KNN=',trACC_KNN)
print('Test Accuracy for KNN=', tesACC_KNN)


# In[113]:


#Calculate accuracy

# Make predictions on the test set
import pandas as pd
#x_test_c = x_test.copy(order='C')  # Copy and enforce C-contiguous memory layout
y_pred = KNN.predict(x_test)




# In[114]:


#Evaluating of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
 
#Calculate accuracy

# Make predictions on the test set
y_pred = KNN.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Prediction Accuracy for KNN :", accuracy)

# Calculate F1 score
f1=f1_score(y_test, y_pred)
print("F1 Score(KNN):", f1)
print("Length of y_tes:", len(y_test))
print("Length of y_pred:", len(y_pred))
    
    
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision(KNN):", precision)
print("Recall:", recall)
 
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (KNN):")
print(conf_matrix)


# In[115]:


# 3. Calculate false positive rate (FPR) and true positive rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 4. Calculate area under the ROC curve (AUC)
auc = roc_auc_score(y_test, y_pred)

# 5. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'KNN (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[45]:


# NB classifier
NB=nb.GaussianNB()
NB.fit(x_train,y_train)
trACC_NB=NB.score(x_train,y_train)
tesACC_NB=NB.score(x_test,y_test)
print('Train_Accuracy for NB=', trACC_NB)
print('Test_Accuracy for NB=', tesACC_NB)


# In[107]:


#Evaluating of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
 
#Calculate accuracy

# Make predictions on the test set
y_pred = NB.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Prediction Accuracy for NB", accuracy)

# Calculate F1 score
f1=f1_score(y_test, y_pred)
print("F1 Score(NB):", f1)
print("Length of y_tes:", len(y_test))
print("Length of y_pred:", len(y_pred))
    
    
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision(NB):", precision)
print("Recall:", recall)
 
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (NB):")
print(conf_matrix)


# In[ ]:


#ROC for NB


# In[49]:


from sklearn.metrics import roc_curve, auc
# Get predicted probabilities for the positive class
y_prob = NB.predict_proba(X_test)[:, 1]

# Calculate the false positive rate (FPR) and true positive rate (TPR) for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

#ploting ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[32]:





# In[33]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[34]:


#Evaluating of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
 
#Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1 score
f1=f1_score(y_test, y_pred)
print("F1 Score:", f1)
print("Length of y_tes:", len(y_test))
print("Length of y_pred:", len(y_pred))
    
    
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
 
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:


#SVM


# In[119]:


# Divide our dataset to train and test
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import sklearn.model_selection as ms 
import sklearn.svm as sv 

x_train,x_test, y_train,y_test=ms.train_test_split(x,y, train_size=0.2, random_state=42)


# In[136]:


# SVM classification
SVM=sv.SVC(kernel='rbf')
SVM.fit(x_train,y_train)
trACC_SVM=clsfi.score(x_train,y_train)
tesACC_SVM=clsfi.score(x_test,y_test)
print('Train Accuracy=',trACC_SVM)
print('Test Accuracy=',tesACC_SVM)


# In[131]:


#Evaluating of the model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
 
# Make predictions on the test set
y_pred = SVM.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Prediction Accuracy for SVM:", accuracy)

# Calculate F1 score
f1=f1_score(y_test, y_pred)
print("F1 Score_SVM:", f1)
print("Length of y_tes:", len(y_test))
print("Length of y_pred:", len(y_pred))
    
    
# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision_SVM:", precision)
print("Recall:", recall)
 
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix_SVM:")
print(conf_matrix)


# In[133]:


trainACC_SVM=[]
testACC_SVM=[]
kernel=[]

for i in ['poly', 'linear', 'sigmoid', 'rbf']:
    clsfi=sv.SVC(kernel=i, degree=2)
    clsfi.fit(x_train,y_train)
    trainACC_SVM.append(clsfi.score(x_train,y_train))
    testACC_SVM.append(clsfi.score(x_test,y_test))
    kernel.append(i)
print(trACC_SVM)
print(tesACC_SVM)    

plt.figure()
plt.plot(trainACC_SVM,label='Train', marker='o')
plt.plot(testACC_SVM,label='Test', marker='o')
plt.xticks([0, 1, 2, 3], kernel)
plt.xlabel('kernel')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# In[137]:


# import module
from tabulate import tabulate
 
# assign data
myresults = [
    ['', 'DT', 'KNN', 'NB','SVM'],
    ['Train', trACC_DT, trACC_KNN, trACC_NB,trACC_SVM],
    ['Test', tesACC_DT, tesACC_KNN, tesACC_NB,tesACC_SVM]
]
 
# display table
print(tabulate(myresults))


# In[ ]:





# In[ ]:




