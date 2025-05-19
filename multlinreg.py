import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklean.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn:linear_model import LinearRegression, Ridge, Lasso
from sklran.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")

#Laden des Datensatzes -- Load Data and Preprocess
df = pd.read_excel('C:/Dokumente/Spyder-AufG/multlinreg.xlsx')
df = df.dropna()

#Create Dummy Variables -- Dummy Variablen erstellen
if 'gender' in df.columns:
  df = pd.get_dummies(df, columns=['gender_M'], drop_first=True)

#Define features and targets -- Targets und Features definieren
target = 'hearing_loss'
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

#Multicolinearity Check: Heatmap and VIF
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.itle("Feature Correlation Heatmap")
plt.show()

#Variance Inflation Factor (VIF)
vif_data['feature'] = X.columns
vif_data['VIF] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#Train-test split and scaling 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.7, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform

