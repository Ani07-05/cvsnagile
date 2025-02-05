# %%
# d:\cvsnagile\Lung Cancer classification.ipynb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("lungcancer.csv")
df.head()

# %%
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['GENDER']=encoder.fit_transform(df[['GENDER']])

# %%
df['LUNG_CANCER']=encoder.fit_transform(df[['LUNG_CANCER']])

# %%
df.head()

# %%
X=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# %%
model=GaussianNB()


# %%
model.fit(X_train,y_train)

# %%
y_pred=model.predict(X_test)
y_pred

# %%
#accuracy

print(model.score(X_test, y_test))

# %%
accuracy_NB=accuracy_score(y_test,y_pred)
print(accuracy_NB)

# %%
#Decision tree

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
model1=DecisionTreeClassifier()

# %%
model1.fit(X_train,y_train)

# %%
y_pred=model1.predict(X_test)
y_pred

# %%
accuracy_DT=accuracy_score(y_test,y_pred)
print(accuracy_DT)

# %%
#Random Forest

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model2=RandomForestClassifier()

# %%
model2.fit(X_train,y_train)

# %%
y_pred=model2.predict(X_test)
y_pred

# %%
accuracy_RF=accuracy_score(y_test,y_pred)
print(accuracy_DT)

# %%
#SVM

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# %%
from sklearn.svm import SVC

# %%
model3=SVC()

# %%
model3.fit(X_train,y_train)

# %%
y_pred=model3.predict(X_test)
y_pred

# %%
accuracy_SVM=accuracy_score(y_test,y_pred)
print(accuracy_SVM)

# %%
#KNN

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
model4=KNeighborsClassifier(n_neighbors=3)

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# %%
model4.fit(X_train,y_train)

# %%
y_pred=model4.predict(X_test)
y_pred

# %%
accuracy_KNN=accuracy_score(y_test,y_pred)
print(accuracy_KNN)

# %%
acc=[]
for i in range(1,50):
    m=KNeighborsClassifier(n_neighbors=i)
    m.fit(X_train,y_train)
    y_pred=m.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
plt.plot(acc)

# %%
acc[5]

# %%
model4=KNeighborsClassifier(n_neighbors=5)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)
accuracy_KNN=accuracy_score(y_test,y_pred)

# %%
#Logistic Resgression

# %%
from sklearn.linear_model import LogisticRegression

# %%
model5=LogisticRegression()

# %%
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)

# %%
model5.fit(X_train,y_train)

# %%
y_pred=model5.predict(X_test)
y_pred

# %%
accuracy_LR=accuracy_score(y_test,y_pred)
print(accuracy_LR)

# %%
#plot

# %%
y_plot=[accuracy_NB,accuracy_DT,accuracy_RF,accuracy_SVM,accuracy_KNN,accuracy_LR]
X_plot=['NB','DT','RF','SVM','KNN','LR']


# %%
plt.plot(X_plot, y_plot, 'o', color='blue') 
plt.plot(X_plot, y_plot)  
plt.grid(True)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()