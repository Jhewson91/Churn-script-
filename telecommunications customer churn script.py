import matplotlib
import pandas as pd
import matplotlib.pyplot as plt  # For basic data visualization
import numpy as np               # For mathematical operations
import seaborn as sns            # For advanced visualisation
from sklearn.model_selection import train_test_split
# For feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
# For rebalancing features
from imblearn.over_sampling import SMOTE
import warnings
# Model building and hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# Performance of Models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Import the telecommunications dataset
tel_data = pd.read_csv(
    'C:/Users/jhews/OneDrive/Documents/Data Analytics/Portfolio/Project 6 - Telecommunications churn predictor/churn_data.csv')
print(tel_data.head(10))  # check the first 10 records of each column
print(tel_data.columns)

# view the unique values in each column
for column in tel_data.columns:
    print('Column: {} , Unique Values: {}'.format(
        column, tel_data[column].unique()))


# Data Cleaning
# Handling Missing Values

print(tel_data.isnull().any())  # returns boolean (true/false)
print(tel_data.isnull().sum())  # returns sum of missing values per column

# transform Total Charges to numeric from object
print(tel_data['TotalCharges'].describe())  # descriptive summary
tel_data["TotalCharges"] = pd.to_numeric(
    tel_data["TotalCharges"], errors='coerce')
print(tel_data['TotalCharges'].describe())  # descriptive summary

print(tel_data.isnull().any())  # returns boolean (true/false)
print(tel_data.isnull().sum())  # returns sum of missing values per column

# Drop null values from dataset
new_tel_data = tel_data.dropna()
# Check null values have been removed
print(new_tel_data.isnull().any())

# Handling duplicated values
print("Duplicated values: ", new_tel_data.duplicated())

# Reformatting Columns
new_tel_data.columns = new_tel_data.columns.str.capitalize()
print(new_tel_data.columns)


# Detecting Outliers

# Tenure boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=new_tel_data['Tenure']).set(
    title='Boxplot showing summary of Tenure')
plt.show()

# Monthly charges boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=new_tel_data['Monthlycharges']).set(
    title='Boxplot showing summary of Monthly Charges')
plt.show()

# Total charges boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=new_tel_data['Totalcharges']).set(
    title='Boxplot showing summary of Total Charges')
plt.show()

# Preliminary Exploration - numeric variables

# Tenure
print(tel_data['tenure'].describe())  # descriptive summary
# "Tenure" Histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(tel_data['tenure'], color=['turquoise'], edgecolor='grey')
plt.xlabel("Tenure (Months)")
plt.ylabel("Frequency")
plt.title("Frequency of Customers by Tenure")
plt.show()

# monthly charges
print(tel_data["MonthlyCharges"].describe())
# monthly charges histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(tel_data['MonthlyCharges'], color=['turquoise'], edgecolor='grey')
plt.xlabel("MonthlyCharges ($)")
plt.ylabel("Frequency")
plt.title("Frequency of monthly charges (US$)")
plt.show()

# monthly charges boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=tel_data['MonthlyCharges']).set(
    title='Boxplot showing summary of Monthly Charges')
plt.show()

# Preliminary Exploration - Categorical variables
# Gender
fig = new_tel_data['Gender'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# SeniorCitizen
fig = new_tel_data['Seniorcitizen'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of SeniorCitizen')
plt.xlabel('Senior Citizen - (1 = Yes, 0 = No)')
plt.ylabel('Frequency')
plt.show()

# Partner
fig = new_tel_data['Partner'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Partners')
plt.xlabel('Partner')
plt.ylabel('Frequency')
plt.show()

# Dependents
fig = new_tel_data['Dependents'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Dependents')
plt.xlabel('Dependents')
plt.ylabel('Frequency')
plt.show()

# Customer Churn
fig = new_tel_data['Churn'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency Customer Churn')
plt.xlabel('Was there churn over the last month')
plt.ylabel('Frequency')
plt.show()

# Contract
fig = new_tel_data['Contract'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Contract Types')
plt.xlabel('Contract Type')
plt.ylabel('Frequency')
plt.show()

# PaymentMethod
fig = new_tel_data['Paymentmethod'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Payment Methods')
plt.xlabel('Paymentmethod')
plt.ylabel('Frequency')
plt.show()

# Conversion of Categorical Variables to Numeric

# Label encoding of binary categorical variables
# create new object and list binary categorical variables
label_encoding = ['Gender', 'Partner', 'Dependents',
                  'Paperlessbilling', 'Phoneservice', 'Churn']
for column in label_encoding:
    if column == "Gender":
        new_tel_data[column] = new_tel_data[column].map(
            {'Female': 1, 'Male': 0})
    else:
        new_tel_data[column] = new_tel_data[column].map(
            {'Yes': 1, 'No': 0})

# one-hot encode remaining categorical variables with multiple levels
one_hot_encode = ['Multiplelines', 'Internetservice', 'Onlinesecurity', 'Onlinebackup',
                  'Deviceprotection', 'Techsupport', 'Streamingtv', 'Streamingmovies', 'Contract', 'Paymentmethod']
new_tel_data = pd.get_dummies(new_tel_data, columns=one_hot_encode)

print(new_tel_data.head(10))


# Normalizing numeric variables using "standardization" technique
# min-max normalization (numeric variables)
min_max_columns = ['Tenure', 'Monthlycharges', 'Totalcharges']

# scale numerical variables using min max scaler
for column in min_max_columns:
    # minimum value of the column
    min_column = new_tel_data[column].min()
    # maximum value of the column
    max_column = new_tel_data[column].max()
    # min max scaler
    new_tel_data[column] = (new_tel_data[column] -
                            min_column) / (max_column - min_column)

print(new_tel_data.describe())

# Feature Selection
# Drop CustomerID, irrelvant
new_tel_data.drop(columns='Customerid', inplace=True)
# Select independent variables
X = new_tel_data.drop(columns='Churn')
# Select dependent variable
y = new_tel_data.loc[:, 'Churn']

# View X and y
print("Independent Variables: ", X.columns)
print("Dependent Variable: ", y.name)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=100, test_size=0.3, shuffle=True)

# Method 1: Univariate Selection
# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

# Method 2: Extra Tree Classifier
model = ExtraTreesClassifier()
model.fit(X, y)
# use inbuilt class feature_importances of tree based classifiers
print(model.feature_importances_)
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# Method 3: Correlation Analysis
corr = new_tel_data.corr()[['Churn']]
print(corr)
# Correlogram heatmap of correlation between "Churn" and other variables
sns.heatmap(corr, annot=True)
plt.show()

# Correlation with output variable
cor_target = abs(corr['Churn'])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.2]
print("Absolute Correlations above 0.2: \n", relevant_features)

# Check for multicollinearity between features
# Create object "X" with independent variables
X = new_tel_data[['Tenure', 'Internetservice_Fiber optic', 'Onlinesecurity_No', 'Techsupport_No',
                  'Contract_Month-to-month', 'Contract_Two year', 'Paymentmethod_Electronic check']]
vif_data = pd.DataFrame()
vif_data["Independent Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print("VIF in Selected Features: \n", vif_data)


# Determining Optimal Model Features using Lasso Regularization:
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" % reg.score(X, y))
coef = pd.Series(reg.coef_, index=X.columns)
print("Lasso picked " + str(sum(coef != 0)) +
      " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Feature importance using Lasso Model")
plt.show()

# Define selected features for ML models:
X = new_tel_data[['Tenure', 'Internetservice_Fiber optic', 'Onlinesecurity_No', 'Techsupport_No',
                  'Contract_Month-to-month', 'Contract_Two year', 'Paymentmethod_Electronic check']]
y = new_tel_data['Churn']

new_tel_data.info()

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print(Counter(y))  # summarize class distribution

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=100, test_size=0.3)

# Model 1: Logistic Regression
warnings.filterwarnings('ignore')
log_reg_model = LogisticRegression()
# Hyperparameter Tuning with GridSearchCV
grid_values_1 = {
    'penalty': ['l1', 'l2'],
    'C': [-1, 0.001, 0.01, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}
grid_search = GridSearchCV(log_reg_model, param_grid=grid_values_1)
grid_search.fit(X_train, y_train)
# Print out best parameters
print("Optimal Parameters :", grid_search.best_params_)  # Print best parameters
print("Accuracy :", grid_search.best_score_)  # Print out Training Accuracy

# Tuned Model Building
log_reg_model = LogisticRegression(
    C=0.01, penalty='l2', solver='liblinear')  # set optimal paramters
log_reg_model.fit(X_train, y_train)  # Train Logistic Regression model
log_reg_predictions = log_reg_model.predict(
    X_test)  # Predict with Logistic Regression model
# Measure performance of the model
print(classification_report(y_test, log_reg_predictions))


# Model 2: Decision Tree
dec_tree_model = DecisionTreeClassifier()
# Hyperparameter Tuning with GridSearchCV
grid_values_2 = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10, 12]
}
grid_search = GridSearchCV(dec_tree_model, param_grid=grid_values_2)
grid_search.fit(X_train, y_train)
# Print out best parameters
print("Optimal Parameters :", grid_search.best_params_)  # Print best parameters
print("Accuracy :", grid_search.best_score_)  # Print out Training Accuracy

# Tuned model building
dec_tree_model = DecisionTreeClassifier(
    criterion='entropy', max_depth=10)  # set optimal paramters
dec_tree_model.fit(X_train, y_train)  # Train decision tree model
dec_tree_predictions = dec_tree_model.predict(
    X_test)  # predict with decision tree
# Measure performance of the model
print(classification_report(y_test, dec_tree_predictions))


# Model 3: Naive Bayes
naive_bayes_model = GaussianNB()
# Hyperparameter Tuning with GridSearchCV
grid_values_3 = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
grid_search = GridSearchCV(naive_bayes_model, param_grid=grid_values_3)
grid_search.fit(X_train, y_train)
# Print out best parameters
print("Optimal Parameters :", grid_search.best_params_)  # Print best parameters
print("Accuracy :", grid_search.best_score_)  # Print out Training Accuracy

# Tuned model building
naive_bayes_model = GaussianNB(var_smoothing=0.1)  # set optimal paramters
naive_bayes_model.fit(X_train, y_train)  # Train naive bayes model
naive_bayes_predictions = naive_bayes_model.predict(
    X_test)  # predict with naive bayes
# Measure performance of the model
print(classification_report(y_test, naive_bayes_predictions))

print("Classifier Algorithm Results: \n")
print("Model 1: Logisitic Regression - Accuracy: ",
      accuracy_score(y_test, log_reg_predictions))
print("Model 2: Decision Tree - Accuracy: ",
      accuracy_score(y_test, dec_tree_predictions))
print("Model 3: Naive Bayes - Accuracy: ",
      accuracy_score(y_test, naive_bayes_predictions))

# Confusion Matrix of Decision Tree
Decision_Tree_confusion_matrix = confusion_matrix(y_test, dec_tree_predictions)
display = ConfusionMatrixDisplay(
    confusion_matrix=Decision_Tree_confusion_matrix, display_labels=dec_tree_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
display.plot()
plt.show()
