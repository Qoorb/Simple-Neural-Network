class Layer:
    def __init__(self, FunctionActivation, CountNeurons):
        self.func = FunctionActivation
        self.qneurons = CountNeurons
        # self.value = [] #TODO

    # Getter for func
    @property
    def function(self):
        return self.func
    
    # Setter for func
    @function.setter
    def function(self, f):
        self.func = f

    # Getter for neurons
    @property
    def neurons(self):
        return self.qneurons
    
    # Setter for neurons
    @neurons.setter
    def neurons(self, n):
        self.qneurons = n