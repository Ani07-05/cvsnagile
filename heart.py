# -*- coding: utf-8 -*-
"""Copy of Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cItv_bN7svGqZDnYl-W9zcs-BYw5DlHp
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("heart.csv")
df.head()
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy:{acc1:.5f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("heart.csv")


x = df.drop('target', axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


acc2 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc2:.5f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("heart.csv")


x = df.drop('target', axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


acc3 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc3:.5f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("heart.csv")


x = df.drop('target', axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = SVC(kernel='linear', random_state=42)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


acc4 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc4:.5f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
acc5 = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc5:.5f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("heart.csv")
df.head()
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
acc6 = accuracy_score(y_test, y_pred)
print(f"Log Accuracy: {acc6:.5f}")

import matplotlib.pyplot as plt

accuracy_values = [acc1, acc2, acc3, acc4, acc5, acc6]


models = ["Naïve Bayes", "Random Forest", "Decision Tree", "SVM", "KNN", "Log Reg"]

plt.figure(figsize=(8, 5))
plt.plot(models, accuracy_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)


plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")
plt.ylim(0.75, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)


plt.show()