class Counter:
    def __init__(self, value=0):
        self.value = value
        self.n = 0

    def addvalue(self, value):
        self.value += value
        self.n += 1

    def getvalue(self):
        mean_value = self.value/self.n
        self.resetvalue()
        return mean_value

    def resetvalue(self):
        self.value = 0
        self.n = 0

    def __add__(self, other):
        self.addvalue(other)
        return self

class StdLogger:
    def __init__(self):
        pass

    def log(self, s):
        print(s)

def clip_gradient(optim_host_gradient, clip_gradient=1.0):
    optim_host_gradient = np.maximum(optim_host_gradient, -clip_gradient)
    optim_host_gradient = np.minimum(optim_host_gradient, clip_gradient)
    return optim_host_gradient