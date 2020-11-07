import numpy as np
from perceptron import Perceptron

class MLP:
    def __init__(self,input_units):
        self.hidden_layer = []
        for i in range(4):
            self.hidden_layer.append(Perceptron(input_units))
        self.output_neuron = Perceptron(4)
    
    def forward_step(self, input):
        from_hidden = np.zeros(4)
        for i, perceptron in zip(range(4), self.hidden_layer):
            from_hidden[i] = perceptron.forward_step(input)
        #print("from_hidden",from_hidden)
        output = self.output_neuron.forward_step(from_hidden) 
        return output
    
    def backprop_step(self, output, target):
        pre_delta_loss = - (target - output)
        delta_output, weights_output = self.output_neuron.update(pre_delta_loss)
        #print("backprob",delta_output, weights_output)
        for perceptron, i in zip(self.hidden_layer, range(4)):
            perceptron.update(delta_output * weights_output[i])