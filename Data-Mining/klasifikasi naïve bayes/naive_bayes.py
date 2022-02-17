from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB  # pip install naive-bayes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # pip install scikit-learn
import numpy as np  # pip install numpy
import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas

# Import Dataset
dataset = pd.read_csv(
    'E:/Portofolio/Python/Data Mining/klasifikasi naïve bayes/dataset/Dataset.csv')  # Tempat penyimpanan File
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Spliting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9)

# Feature Scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set result
y_pred = classifier.predict(X_test)

# Making the Confusion  Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualising the Training set Result
X_set, y_set, = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Training Social Network)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Training set Result
X_set, y_set, = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Test set Social Network)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
