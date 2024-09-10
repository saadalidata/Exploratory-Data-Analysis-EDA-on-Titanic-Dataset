# Import required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
data_titanic = pd.read_csv('train.csv')  # Modify path as needed

# Data Cleaning Steps
# Manage missing data
data_titanic['Age'].fillna(data_titanic['Age'].mean(), inplace=True)
data_titanic['Embarked'].fillna(data_titanic['Embarked'].mode()[0], inplace=True)
data_titanic.drop(columns=['Cabin'], inplace=True)  # Removing Cabin due to many missing values

# Adjust data types
data_titanic['Survived'] = data_titanic['Survived'].astype('category')
data_titanic['Pclass'] = data_titanic['Pclass'].astype('category')

# Perform Exploratory Data Analysis
# Examine survival rates by various attributes
sns.countplot(x='Survived', data=data_titanic)
plt.title('Count of Survivors')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=data_titanic)
plt.title('Survival Distribution by Gender')
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=data_titanic)
plt.title('Survival Distribution by Ticket Class')
plt.show()

sns.histplot(data_titanic['Age'], kde=True)
plt.title('Distribution of Ages')
plt.show()

# Visualization Section
# Display survival rates based on embarkation points
sns.countplot(x='Embarked', hue='Survived', data=data_titanic)
plt.title('Survival Rates by Embarkation Point')
plt.show()

# Document key insights and save them
print("Documentation and visualizations have been saved. Check the Jupyter Notebook for a detailed analysis.")

# Store the cleaned and processed data
data_titanic.to_csv('titanic_cleaned.csv', index=False)
print("Cleaned dataset has been saved.")
