import os
#os.chdir('../')
from redunet import ReduNetVector
import examples.utils_example as ue
import plot

## Data
dataset = 3  # can be 1 to 8
train_noise = 0.1
test_noise = 0.1
train_samples = 400
test_samples = 400

## Model
num_layers = 200  # number of redunet layers
eta = 0.5
eps = 0.01
lmbda = 200

X_train, y_train, num_classes = ue.generate_3d(dataset, train_noise, train_samples) # train
X_test, y_test, num_classes = ue.generate_3d(dataset, test_noise, test_samples) # test

net = ReduNetVector(num_classes, num_layers, 
                    X_train.shape[1], eta=eta, eps=eps, lmbda=lmbda)
Z_train = net.init(X_train, y_train)

ue.plot_loss_mcr(net.get_loss())
ue.plot_3d(X_train, y_train, 'X_train') 
ue.plot_3d(Z_train, y_train, 'Z_train') 

Z_test, Z_test_interm = net(X_test, return_inputs=True)

ue.plot_3d(X_test, y_test, 'X_test')

for i in range(len(Z_test_interm)):
    ue.plot_3d(Z_test_interm[i], y_test, 'Z_test_interm_{}'.format(i))

ue.plot_3d(Z_test, y_test, 'Z_test')
