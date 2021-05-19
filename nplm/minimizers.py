import numpy as np

from .neurnetmodel import Network, Ilogit_and_KL
from .data_manipulation import one_hot_encode_matrix

def train_model(N, X, Y, dict_size, batch_size=100, nb_epochs=20, alpha=1e-5, logs=1):
    """
    Train the neural network using stochastic gradient descent
    with the specified batch size
    """
    n = X.shape[0]
    Ix = np.arange(n)
    theta = N.get_params()
    cost_list = []
    for i in range(nb_epochs):
        np.random.shuffle(Ix)
        current_end = 0
        epoch_cost = 0
        while current_end < n:
            start = current_end
            end = start + batch_size
            current_end = end
            X_batch = X[Ix[start:end], :]
            X_batch = one_hot_encode_matrix(X_batch, dict_size)
            Y_temp = Y[Ix[start:end], :]
            Y_temp = one_hot_encode_matrix(Y_temp, dict_size)
            Y_batch = np.concatenate(Y_temp).reshape(Y_temp.shape[1], -1, order='F')
            N_a_batch = Network([N, Ilogit_and_KL(Y_batch)])
            cost = N_a_batch.forward(X_batch)
            epoch_cost += cost
            grad = N_a_batch.backward(None)[0]
            theta = theta - alpha * grad
            N_a_batch.set_params(theta)
        if logs==1:
            print("Epoch {} completed. Cost function = {}".format(i, epoch_cost))
        cost_list.append(epoch_cost)
    return cost_list

