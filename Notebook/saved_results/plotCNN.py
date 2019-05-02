import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('cnn1Data.csv')
data2 = pd.read_csv('cnn1.5Data.csv')
data3 = pd.read_csv('cnn1.5Data4.csv')
data4 = pd.read_csv('cnn2Data4.csv')

xticks1 = range(0,1+1)
xticks2 = range(0,4+1)

plt.plot(np.array(range(0,data1.shape[0])) / data1.shape[0], data1['accuracy'])
plt.plot(np.array(range(0,data2.shape[0])) / data1.shape[0], data2['accuracy'])
plt.xticks(xticks1, xticks1)
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.ylim([0, 1])
plt.legend(['1 to D CNN', '3-3 to D CNN'])
plt.title('Simple CNN architectures')
plt.show()

plt.plot(np.array(range(0,data1.shape[0])) / data1.shape[0], data1['accuracy'])
plt.plot(np.array(range(0,data2.shape[0])) / data1.shape[0], data2['accuracy'])
plt.plot(np.array(range(0,data3.shape[0])) / data1.shape[0], data3['accuracy'])
plt.plot(np.array(range(0,data4.shape[0])) / data1.shape[0], data4['accuracy'])
plt.xticks(xticks2, xticks2)
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.ylim([0, 1])
plt.title('More complex architectures and more epochs')
plt.legend(['1 CNN', '3-3 CNN', '3-3 CNN (4 EP)', '32-64-128 CNN (4 EP)'])
plt.show()
