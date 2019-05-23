import numpy as np
from copy import deepcopy as copy
def evaluate(x,p):
    pa =copy(p)
    print(pa.shape)
    w1 = pa[:,:943*1886]
    w1 = w1.reshape((943,1886))
    pa = pa[:,943*1886:]
    b1 = np.array((pa[:,:1886]))
    pa = pa[:,1886:]
    w2 = np.array(pa[:,:1886*4760])
    w2 = w2.reshape((1886,4760))
    pa = pa[:,1886*4760:]
    b2 = np.array(pa)

    h1 = np.tanh(np.dot(x,w1) + b1)
    out = np.dot(h1,w2) + b2
    return out

def fitness(y_i,y_i_hat):
    return np.sum(np.square(y_i-y_i_hat))

def main():
    X_tr = np.load('bgedv2_X_tr_float64.npy')
    print(X_tr.shape)
    Y_tr = np.load('bgedv2_Y_tr_0-4760_float64.npy')
    print(Y_tr.shape)
    Y_tr_target = np.array(Y_tr)
    '''X_va = np.load('bgedv2_X_va_float64.npy')
    Y_va = np.load('bgedv2_Y_va_0-4760_float64.npy')
    Y_va_target = np.array(Y_va)
    X_te = np.load('bgedv2_X_te_float64.npy')
    Y_te = np.load('bgedv2_Y_te_0-4760_float64.npy')
    Y_te_target = np.array(Y_te)

    X_1000G = np.load('1000G_X_float64.npy')
    Y_1000G = np.load('1000G_Y_0-4760_float64.npy')
    Y_1000G_target = np.array(Y_1000G)
    X_GTEx = np.load('GTEx_X_float64.npy')
    Y_GTEx = np.load('GTEx_Y_0-4760_float64.npy')
    Y_GTEx_target = np.array(Y_GTEx)
    '''
    '''
    Initialize QPSO here. dimension of qpso particles is no of particles * 10762504
    No constraints, initialize in range (0,1)
    
    '''
    MSE = [] #Mean squared errors
    for epoch in range(0,1000):
        errors = []
        for data in range(X_tr.shape[0]):
            X_i = X_tr[data,:]
            Y_i = Y_tr[data,:]
            # pa = potential solution
            Y_i_hat = evaluate(X_i,pa)
            fitness(Y_i,Y_i_hat) #fitness function to optimize

            #one iteration of QPSO

            errors.append(fitness(Y_i,evaluate(X_i,g_best)))
        MSE.append(sum(errors)/X_tr.shape[0])
    
    #plot MSE across epochs

        



            
main()

#pa = np.random.rand(1,10762504)
#x = np.random.rand(1,943)
#print(evaluate(x,pa))