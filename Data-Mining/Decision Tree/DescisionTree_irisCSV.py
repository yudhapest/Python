import numpy as np
import pandas as pd
from sklearn import tree

irisDataset = pd.read_csv(
    'E:/Portofolio/Python/Data Mining/Decision Tree/dataset/Dataset.csv', delimiter=';', header=0)
irisDataset.head()

irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]
irisDataset.head()

print(irisDataset)

# menghapus kolom id
irisDataset = irisDataset.drop(labels="Id", axis=1)
print(irisDataset)

# mengubah data frame
irisDataset = irisDataset.to_numpy()
print(irisDataset)

# membagi dataset > 80baris data untuk training dan 20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40, :], irisDataset[50:90, :]),
                              axis=0)
dataTesting = np.concatenate((irisDataset[40:50, :], irisDataset[90:100, :]),
                             axis=0)

print(dataTraining)
len(dataTraining)

print(dataTesting)
len(dataTesting)

inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]
print(labelTraining)
len(labelTraining)

# mendefinisikan DT calssifier
model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)

# memprediksi input data testing
hasilPrediksi = model.predict(inputTesting)
print("Label Sebenarnya : ", labelTesting)
print("HasilPrediksi: ", hasilPrediksi)

prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi Benar :", prediksiBenar, "data")
print("prediksi salah :", prediksiSalah, "data")
print("Akurasi :", prediksiBenar/(prediksiBenar+prediksiSalah)*100, "%")
