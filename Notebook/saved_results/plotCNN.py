import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('cnnTanhValData.csv')
data2 = pd.read_csv('cnnMSETanhValData.csv')
data3 = pd.read_csv('cnnRELUValData.csv')
data4 = pd.read_csv('cnnMSERELUValData.csv')
data5 = pd.read_csv('cnnLeakyRELUValData.csv')
data6 = pd.read_csv('cnnMSELeakyRELUValData.csv')

data1b = pd.read_csv('cnnTanhData.csv')
data2b = pd.read_csv('cnnMSETanhData.csv')
data3b = pd.read_csv('cnnRELUData.csv')
data4b = pd.read_csv('cnnMSERELUData.csv')
data5b = pd.read_csv('cnnLeakyRELUData.csv')
data6b = pd.read_csv('cnnMSELeakyRELUData.csv')

xstuff = np.array(range(0,data1b.shape[0])) * 16 / float(data1b.shape[0])
skipper = np.array(range(0,data1b.shape[0], int(data1b.shape[0]/16)-1))
# skipper = np.array(range(0,data1b.shape[0], 1))

plt.plot(xstuff[skipper], np.array(data1b['accuracy'])[skipper])
plt.plot(xstuff[skipper], np.array(data2b['accuracy'])[skipper])
plt.plot(xstuff[skipper], np.array(data3b['accuracy'])[skipper])
plt.plot(xstuff[skipper], np.array(data4b['accuracy'])[skipper])
plt.plot(xstuff[skipper], np.array(data5b['accuracy'])[skipper])
plt.plot(xstuff[skipper], np.array(data6b['accuracy'])[skipper])
plt.xlabel('Epochs')
plt.ylabel('Training Accuarcy')
plt.xlim([-0.1, 16.1])
plt.ylim([0, 1])
plt.title('CNN training accuracy during a few epochs of training')
plt.legend(['Tanh', 'Tanh + MSE', 'relu', 'relu + MSE', 'leaky relu', 'leaky relu + MSE'])
plt.show()

xticks = range(0,16+1)

plt.plot(np.array(range(1,1+data1.shape[0])), data1['accuracy'])
plt.plot(np.array(range(1,1+data2.shape[0])), data2['accuracy'])
plt.plot(np.array(range(1,1+data3.shape[0])), data3['accuracy'])
plt.plot(np.array(range(1,1+data4.shape[0])), data4['accuracy'])
plt.plot(np.array(range(1,1+data5.shape[0])), data5['accuracy'])
plt.plot(np.array(range(1,1+data6.shape[0])), data6['accuracy'])
plt.xticks(xticks, xticks)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuarcy')
plt.xlim([-0.1, 16.1])
plt.ylim([0, 1])
plt.title('CNN validation accuracy after a few epochs of training')
plt.legend(['Tanh', 'Tanh + MSE', 'relu', 'relu + MSE', 'leaky relu', 'leaky relu + MSE'])
plt.show()

print(max(data1['accuracy']))
print(max(data2['accuracy']))
print(max(data3['accuracy']))
print(max(data4['accuracy']))
print(max(data5['accuracy']))
print(max(data6['accuracy']))

print()

print(max(data1b['accuracy']))
print(max(data2b['accuracy']))
print(max(data3b['accuracy']))
print(max(data4b['accuracy']))
print(max(data5b['accuracy']))
print(max(data6b['accuracy']))
