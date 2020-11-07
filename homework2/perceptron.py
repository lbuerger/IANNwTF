import numpy as np

class Perceptron:
    def __init__(self, input_units):
        self.inputs = None
        self.alpha = 1 # learning rate
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn(1)
        self.drive = None
        
    """https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python"""    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    """https://stackoverflow.com/questions/10626134/derivative-of-sigmoid"""
    def sigmoidprime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward_step(self,input):
        drive = np.dot(self.weights, input) + self.bias
        self.drive = drive
        self.inputs = input
        return self.sigmoid(drive)
    
    def training_step(self, input, label):
        # 1. forward step
        output = self.forward_step(input)
        # 2. pre_delta for the parameter updates
        pre_delta = -(label - output) 
        self.update(pre_delta)
        
        
    def update(self, pre_delta):
        delta = pre_delta * self.sigmoidprime(self.drive)
        self.bias = self.bias - self.alpha * delta * 1 
        old_weights = self.weights[:]
        self.weights = self.weights - self.alpha * delta * self.inputs
        #print(old_weights, self.weights)
        #print(self.inputs.shape,self.weights.shape)
        return delta, old_weights