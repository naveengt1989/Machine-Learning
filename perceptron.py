import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions
class Perceptron(object):

    def __init__(self, eta, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.w_[1]=1
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


X = np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
Y = np.array([1,1,0,0])
p = Perceptron(epochs=10, eta=1)
p.train(X, Y)

print('Weights: %s' % p.w_)
plot_decision_regions(X, Y, clf=p)
plt.title('Perceptron')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.plot(range(1, len(p.errors_)+1), p.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()
