# Titanic_Classification

# Aim
The aim of this project is to build a model that predicts whether a passenger on the Titanic survived or not based on given features.

# Dataset
The dataset for this project is imported from a CSV file, "tested.csv". The dataset contains information about passengers on the Titanic, including their survival status, class (Pclass), sex (Gender), and age (Age).

# Libraries
The following important libraries were used for this project:

* numpy
* pandas
* matplotlib.pyplot
* seaborn
* sklearn.preprocessing.LabelEncoder
* sklearn.model_selection.train_test_split
* sklearn.linear_model.LogisticRegression

# Data Exploration and Data Preprocessing
1. The dataset was loaded as a DataFrame with pandas, and its shape and a preview of the first 10 rows were displayed with df.shape and df.head(10).
2. To acquire an overview of the data, descriptive statistics for the numerical columns were displayed using df.describe().
3. sns.countplot(x=df['Survived']) was used to visualise the number of passengers who survived and those who did not.
4. The number of survivors was plotted against the Pclass using sns.countplot(x=df['Survived'], hue=df['Pclass']).
5. sns.countplot(x=df['Sex'], hue=df['Survived']) was used to visualise the number of survivors by gender.
6. Using df.groupby('Sex')[['Survived']], the survival rate by gender was calculated and displayed.mean().
7. Using LabelEncoder from sklearn.preprocessing, the 'Sex' column was converted from categorical to numerical values.
8. Non-required columns such as 'Age' were removed from the DataFrame after encoding the 'Sex' column.

# Model Training 
1. Using relevant columns from the DataFrame, the feature matrix X and target vector Y were generated.
2. Using train_test_split from sklearn.model_selection, the dataset was divided into training and testing sets.
3. Using LogisticRegression from sklearn.linear_model, a logistic regression model was created and trained on the training data.

# Model Prediction
1. The model was used to forecast the survival status of test passengers.
2. Log.predict(X_test) was used to print the predicted results.
3. Y_test was used to print the actual target values in the test set.
4. Log.predict([[2, 1]]) with Pclass=2 and Sex=Male (1) was used to make a sample prediction.
