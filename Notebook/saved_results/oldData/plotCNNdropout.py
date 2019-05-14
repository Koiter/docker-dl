import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('cnn1Data4.csv')
data2 = pd.read_csv('cnn1DropoutData4.csv')
data3 = pd.read_csv('cnn2Data4.csv')
data4 = pd.read_csv('cnn2DropoutData4.csv')

xticks1 = range(0,4+1)

plt.plot(np.array(range(0,data3.shape[0])) * 4 / data1.shape[0], data3['accuracy'])
plt.plot(np.array(range(0,data4.shape[0])) * 4 / data1.shape[0], data4['accuracy'])
plt.plot(np.array(range(0,data1.shape[0])) * 4 / data1.shape[0], data1['accuracy'])
plt.plot(np.array(range(0,data2.shape[0])) * 4 / data1.shape[0], data2['accuracy'])
plt.xticks(xticks1, xticks1)
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.ylim([0, 1])
plt.legend(['32-64-128 CNN', '32-64-128 CNN + dropout', '1 CNN', '1 CNN + Dropout'])
plt.title('The effect of dropout on the small and large CNN')
plt.show()
