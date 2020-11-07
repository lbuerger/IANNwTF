import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
#from perceptron import Perceptron

data = np.array([[0,0],[0,1],[1,0],[1,1]])

#targets
t_and = np.array([0,0,0,1])
t_or = np.array([0,1,1,1])
t_nand = np.array([1,1,1,0])
t_nor = np.array([1,0,0,0])
t_xor = np.array([0,1,1,0])


# training

t = t_xor

mlp = MLP(2)

#training loop
epochs = []
accuracies = []
losses = []

for i in range(1000):
    epochs.append(i)
    accuracy_sum = 0
    loss_sum = 0
    for k in range(4):
        output = mlp.forward_step(data[k])
        mlp.backprop_step(output, t[k])
        loss_sum += (t[k]-output)**2
        if output > 0.5:
            output = 1
        else:
            output = 0
        accuracy_sum += int(output == t[k])
    accuracy = accuracy_sum/4
    loss = loss_sum/4
    accuracies.append(accuracy)
    losses.append(loss[0])

print("After {} Epochs: {} loss, {} accuracy".format(len(epochs), losses[-1], accuracies[-1]))
# visualize training

plt.figure()
#print(epochs)
#print(accuracies)
plt.plot(epochs,accuracies,label="accuracy")
plt.plot(epochs,losses,label="loss")
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.ylim([-0.1,1.1])
plt.legend()
plt.title("Ugly but working MLP")
plt.show()