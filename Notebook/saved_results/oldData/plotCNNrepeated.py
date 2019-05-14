import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('cnn1Data4.csv')
data2 = pd.read_csv('cnn1Data4b.csv')
data3 = pd.read_csv('cnn1Data4c.csv')
data4 = pd.read_csv('cnn1Data4d.csv')
data5 = pd.read_csv('cnn1Data4e.csv')

xticks1 = range(0,4+1)

plt.plot(np.array(range(0,data1.shape[0])) * 4 / data1.shape[0], data1['accuracy'])
plt.plot(np.array(range(0,data2.shape[0])) * 4 / data1.shape[0], data2['accuracy'])
plt.plot(np.array(range(0,data3.shape[0])) * 4 / data1.shape[0], data3['accuracy'])
plt.plot(np.array(range(0,data4.shape[0])) * 4 / data1.shape[0], data4['accuracy'])
plt.plot(np.array(range(0,data5.shape[0])) * 4 / data1.shape[0], data5['accuracy'])
plt.xticks(xticks1, xticks1)
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.ylim([0, 1])
plt.title('Differently initialized and trained CNNs')
plt.show()
