
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from xgboost import XGBClassifier
from tabulate import tabulate
import optuna
import time
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Load data
train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')

print(train.head())
print(train.info())
print(train.describe())
print(train.describe(include='object'))
print(train.isnull().sum())

total = train.shape[0]

# Missing value report
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count / total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

print(f"Number of duplicate rows: {train.duplicated().sum()}")

sns.countplot(x=train['class'])
plt.show()

print("Class distribution Training set:")
print(train['class'].value_counts())

# Label Encoding
def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

# Drop constant column
if 'num_outbound_cmds' in train.columns:
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Feature Selection
X_train = train.drop("class", axis=1)
Y_train = train["class"]

rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

selected_features = X_train.columns[rfe.get_support()].tolist()

print("Selected Features:", selected_features)

X_train = X_train[selected_features]

# Scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.transform(test)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X_train, Y_train, train_size=0.70, random_state=2)

# Logistic Regression
clfl = LogisticRegression(max_iter=1200000)
start = time.time()
clfl.fit(x_train, y_train)
print("LR Training time:", time.time() - start)

start = time.time()
clfl.predict(x_train)
print("LR Testing time:", time.time() - start)

lg_model = LogisticRegression(random_state=42)
lg_model.fit(x_train, y_train)
lg_train = lg_model.score(x_train, y_train)
lg_test = lg_model.score(x_test, y_test)

print(f"LR Train Score: {lg_train}")
print(f"LR Test Score: {lg_test}")

# ==== OPTUNA FOR KNN =====
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_knn(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 16)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(objective_knn, n_trials=10)

best_k = study_KNN.best_trial.params['n_neighbors']
KNN_model = KNeighborsClassifier(n_neighbors=best_k)
KNN_model.fit(x_train, y_train)

KNN_train = KNN_model.score(x_train, y_train)
KNN_test = KNN_model.score(x_test, y_test)

print(f"KNN Train Score: {KNN_train}")
print(f"KNN Test Score: {KNN_test}")

# ==== OPTUNA for Decision Tree ====
def objective_dt(trial):
    max_depth = trial.suggest_int("max_depth", 2, 32)
    max_features = trial.suggest_int("max_features", 2, min(10, X_train.shape[1]))
    model = DecisionTreeClassifier(max_features=max_features, max_depth=max_depth)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=30)

best_params = study_dt.best_trial.params
dt = DecisionTreeClassifier(**best_params)
dt.fit(x_train, y_train)

dt_train = dt.score(x_train, y_train)
dt_test = dt.score(x_test, y_test)

print(f"DT Train Score: {dt_train}")
print(f"DT Test Score: {dt_test}")

# Summary Table
data = [
    ["KNN", KNN_train, KNN_test],
    ["Logistic Regression", lg_train, lg_test],
    ["Decision Tree", dt_train, dt_test]
]

col_names = ["Model", "Train Score", "Test Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

# === Cross Validation ===
models = {
    "KNN": KNN_model,
    "LogisticRegression": lg_model,
    "DecisionTree": dt
}

scores = {}
for name, model in models.items():
    scores[name] = {}
    for scorer in ["precision", "recall"]:
        cv = cross_val_score(model, x_train, y_train, cv=10, scoring=scorer)
        scores[name][scorer] = cv.mean() * 100

score_df = pd.DataFrame(scores).T
score_df.plot(kind="bar", figsize=(12, 6), ylim=[80, 100])
plt.show()

# Predictions & Evaluation
preds = {name: model.predict(x_test) for name, model in models.items()}
print("Predictions complete.")

for name in preds:
    print(f"\n=== {name} Confusion Matrix ===")
    print(confusion_matrix(y_test, preds[name]))
    print(classification_report(y_test, preds[name]))


f1s = {name: f1_score(y_test, preds[name]) * 100 for name in models}
pd.DataFrame.from_dict(f1s, orient="index", columns=["F1 Score"]).plot(
    kind="bar", figsize=(8, 5), ylim=[80, 100])
plt.show()
