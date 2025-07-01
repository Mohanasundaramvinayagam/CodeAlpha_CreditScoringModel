#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import confusion_matrix

# E



# In[2]:


data = pd.read_csv('credit_data.csv')
print(data.head())
print(data.info())


# In[3]:


# Check for null values
print(data.isnull().sum())

# Drop or fill nulls
data.fillna(data.mean(), inplace=True)

# Separate features and target
X = data[['income', 'debt', 'payment_history']]
y = data['credit_score']

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


# In[5]:


log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)


# In[6]:


tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)


# In[7]:


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# In[8]:


def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[9]:


evaluate_model(y_test, y_pred_log, "Logistic Regression")
evaluate_model(y_test, y_pred_tree, "Decision Tree")
evaluate_model(y_test, y_pred_rf, "Random Forest")


# In[ ]:





# In[ ]:


log_prob = log_model.predict_proba(X_test)[:, 1]  # probability of class 1 (creditworthy)
rf_prob = rf_model.predict_proba(X_test)[:, 1]    # probability of class 1 (creditworthy)
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.plot(fpr_log, tpr_log, label="Logistic")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1],[0,1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




