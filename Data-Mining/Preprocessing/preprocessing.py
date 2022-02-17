from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # pip install scikit-learn
import numpy as np  # pip install numpy
import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas

"""
Numpy merupakan library python yang digunakan untuk komputasi matrix.
Matplotlib murupakan library python yang digunakan untuk presentasi data berupa grafik atau plot.
Pandas merupakan library python yang digunakan untuk membuat tabel, mengubah dimensi data, mengecek data, dan lain sebagainya. 
"""
dataset = pd.read_csv(
    'E:/Portofolio/Python/Data Mining/Preprocessing/dataset/Data.csv')  # Tempat penyimpanan file
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset)

print("#################################################################")
print(x)
print(y)

print("#################################################################")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

print("#################################################################")
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

print("#################################################################")
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

print("#################################################################")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

print("#################################################################")
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
