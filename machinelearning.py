## Pipeline and Preprocessing

## Train-Test Split
from sklearn.model_selection import train_test_split

## Creating Pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

## Imbalanced pipeline and SMOTE
# from imblearn.pipeline import Pipeline, make_pipeline
# from imblearn.pipeline import make_pipeline

## Sampling Techniques
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

## For preprocessing
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
# Label Encoding
from sklearn.preprocessing import LabelEncoder
# Standardization
from sklearn.preprocessing import StandardScaler
# Normalizaation
from sklearn.preprocessing import MinMaxScaler
# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder

## For missing values
from sklearn.impute import SimpleImputer

## Creating a function transformer
from sklearn.preprocessing import FunctionTransformer

## For Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

## Machine Learning Models

## Classification Models
## Logistic Regression
from sklearn.linear_model import LogisticRegression
## Support Vector Classifiers
from sklearn.svm import SVC
## Decision Tree
from sklearn.tree import DecisionTreeClassifier
## Random Forest
from sklearn.ensemble import RandomForestClassifier
## Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
## K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
## XGboost
from xgboost import XGBClassifier

## Regression Models
## Linear Regression
from sklearn.linear_model import LinearRegression
## Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
## Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

## Cross Validationand Hyperparameter tuning
## Creation of different folds
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


## Cross validation
from sklearn.model_selection import cross_val_score

## Gridsearch CV
from sklearn.model_selection import GridSearchCV

## Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV