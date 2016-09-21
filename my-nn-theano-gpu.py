import theano
import time
import theano.tensor as T
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class NNModel:
    def __init__(self, layers, epsilon = 0.01, reg_lambda = 0.01):
        self.layers = layers # number of nodes in each layer
        self.epsilon = np.float32(epsilon) # learning rate for gradient descent
        self.reg_lambda = np.float32(reg_lambda) # regularization strength        
        
        # Initialize the parameters (W and b) to random values. We need to learn these.        
        #np.random.seed(int(time.time()) % 1000)
        np.random.seed(0)
        hidden_layer_num = len(layers) - 1
        if hidden_layer_num != 2:
            print 'only support 2 hidden layer'
            exit(0)
        self.W1 = theano.shared(np.random.randn(layers[0], layers[1]).astype('float32'))
        self.b1 = theano.shared(np.zeros(layers[1]).astype('float32'))
        self.W2 = theano.shared(np.random.randn(layers[1], layers[2]).astype('float32'))
        self.b2 = theano.shared(np.zeros(layers[2]).astype('float32'))
    
    def train_init(self, train_X, train_y):
        num_class = np.max(train_y) + 1
        num_examples = len(train_X)
        train_y_onehot = np.eye(num_class)[train_y]
        
        # GPU NOTE: Conversion to float32 to store them on the GPU!
        X = theano.shared(train_X.astype('float32'))
        y = theano.shared(train_y_onehot.astype('float32'))
        
        # Forward propagation
        z1 = X.dot(self.W1) + self.b1
        a1 = T.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        y_hat = T.nnet.softmax(z2)
        
        #Prediction
        prediction = T.argmax(y_hat, axis=1)

        #Loss function
        loss_reg = 1. / num_examples * self.reg_lambda / 2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2)))
        loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

        # Gradients
        dW2 = T.grad(loss, self.W2)
        db2 = T.grad(loss, self.b2)
        dW1 = T.grad(loss, self.W1)
        db1 = T.grad(loss, self.b1)

        # Note that we removed the input values because we will always use the same shared variable
        # GPU NOTE: Removed the input values to avoid copying data to the GPU.
        forward_prop = theano.function([], y_hat)
        predict_by_train_X = theano.function([], prediction)
        self.calculate_loss = theano.function([], loss)
        self.gradient_step = theano.function(
            [],
            #profile=True,
            updates=((self.W2, self.W2 - self.epsilon * dW2),
                     (self.W1, self.W1 - self.epsilon * dW1),
                     (self.b2, self.b2 - self.epsilon * db2),
                     (self.b1, self.b1 - self.epsilon * db1)))

    # This function learns parameters for the neural network from training dataset
    # - num_passes: Number of passes through the training data for gradient descent
    # - print_loss: If True, print the loss every 1000 iterations
    def train(self, train_X, train_y, num_passes=20000, print_loss=False):
        self.train_init(train_X, train_y)
        
        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):
            # This will update our parameters Ws and bs!
            self.gradient_step()

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" %(i, self.calculate_loss())

        self.predict_init()

    def predict_init(self):
        X = T.fmatrix('X')

        # Forward propagation
        z1 = X.dot(self.W1) + self.b1
        a1 = T.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        y_hat = T.nnet.softmax(z2)
        
        prediction = T.argmax(y_hat, axis=1)
        
        self.predict_by_X = theano.function([X], prediction)
        
    
    def predict(self, predict_X):
        result = self.predict_by_X(predict_X.astype('float32'))
        return result

def generate_data(random_seed, n_samples):
    np.random.seed(random_seed)
    X, y = datasets.make_moons(n_samples, noise=0.20)
    return X, y

def visualize(X, y, model):
    plt.title("nn_theano_gpu_classification")
    plot_decision_boundary(lambda x:model.predict(x), X, y)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Config:
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength
    layers = [2, 4, 2] # number of nodes in each layer
    num_passes = 20000
    print_loss = True
    random_seed = 6
    num_samples = 5000

X, y = generate_data(Config.random_seed, Config.num_samples)
model = NNModel(Config.layers, Config.epsilon, Config.reg_lambda)

# model.train_init(X, y)
# %timeit model.gradient_step()
start_time = time.time()
model.train(X, y, Config.num_passes, Config.print_loss)
end_time = time.time()
print 'training time : ' + str(end_time - start_time)

visualize(X, y, model)
#model.gradient_step.profile.summary()