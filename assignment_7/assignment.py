# Assignemnt 7
# Tamim Shaban
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('explo.csv')
data
data.head()
data.describe()
data.shape
data.isnull().sum()
students = data.drop(['race/ethnicity', 'parental level of education'], axis=1)
cor = students.corr()
sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
sns.replot(x='math score', y='reading score', hue='gender', data=students)
sns.pairplot(students)
plt.show()