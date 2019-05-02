import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('baseData.csv')
data2 = pd.read_csv('baseDropoutData.csv')
data3 = pd.read_csv('sgdData.csv')

xticks1 = range(0,1+1)
xticks2 = range(0,4+1)

plt.plot(np.array(range(0,data1.shape[0])) / data1.shape[0], data1['accuracy'])
plt.plot(np.array(range(0,data2.shape[0])) / data1.shape[0], data2['accuracy'])
plt.plot(np.array(range(0,data3.shape[0])) / data1.shape[0], data3['accuracy'])
plt.xticks(xticks1, xticks1)
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.ylim([0, 1])
plt.legend(['Base', 'Base + Dropout', 'SGD'])
plt.title('Dense layers with feature extractor')
plt.show()
