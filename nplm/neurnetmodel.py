import numpy as np

class Generic() : 
    def __init__(self) :
        self.nb_params=None # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return None
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=None
        return grad_local,grad_entree
    
class Arctan() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return np.arctan(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        grad_entree = 1/(1 + self.save_X**2) * grad_sortie
        return grad_local,grad_entree

class Tanh() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return np.tanh(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        grad_entree = (1 - np.tanh(self.save_X)**2) * grad_sortie
        return grad_local,grad_entree

class Sigmoid() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return 1 / (1 + np.exp(-X))
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        exp_minusX = np.exp(-self.save_X)
        grad_entree = (exp_minusX / (1 + exp_minusX)**2) * grad_sortie
        return grad_local,grad_entree
    
class RELU() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return X * (X>0)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        X = self.save_X
        grad_entree = (X/np.abs(X)) * (X>0) * grad_sortie
        return grad_local,grad_entree

class ConcatProjections() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return X.reshape(X.shape[0], -1, order='F').T
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        grad_entree = grad_sortie.T.reshape(self.save_X.shape, order='F')
        return grad_local,grad_entree
    
class ABS() : 
    def __init__(self) :
        self.nb_params = 0 # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return np.abs(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        X = self.save_X
        grad_entree = X/np.abs(X) * grad_sortie
        return grad_local,grad_entree

class ProjectVectors() : 
    def __init__(self, n_entree, n_sortie) :
        self.n_entree = n_entree
        self.n_sortie = n_sortie
        self.A = np.random.randn(n_sortie, n_entree)
        self.nb_params = (n_entree) * n_sortie # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        self.A = params.reshape((self.n_sortie, self.n_entree))
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.ravel(self.A)
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return np.matmul(self.A, X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        Xt = np.transpose(self.save_X, (0, 2, 1))
        Xr = np.matmul(grad_sortie, Xt)
        gA = np.sum(Xr, axis=0)
        grad_local = np.ravel(gA)
        grad_entree = None # This is the first layer, no need for this
        return grad_local, grad_entree

class Dense() : 
    def __init__(self, n_entree, n_sortie) :
        self.n_entree = n_entree
        self.n_sortie = n_sortie
        self.A = np.random.randn(n_sortie, n_entree)
        self.b = np.random.randn(n_sortie, 1)
        self.nb_params = (n_entree + 1) * n_sortie # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        end_A = self.n_entree * self.n_sortie
        self.A = params[:end_A].reshape((self.n_sortie, self.n_entree))
        self.b = params[end_A:].reshape((self.n_sortie, 1))
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.concatenate([np.ravel(self.A), np.ravel(self.b)])
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return self.A.dot(X) + np.outer(self.b, np.ones(X.shape[1]))
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        gA = grad_sortie.dot(self.save_X.T)
        gb = np.sum(grad_sortie, axis=1)
        grad_local = np.concatenate([np.ravel(gA), np.ravel(gb)])
        grad_entree = np.dot(self.A.T, grad_sortie)
        return grad_local,grad_entree

class Loss_L2() : 
    def __init__(self, D) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.D = D
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return 0.5 * np.linalg.norm(X - self.D) **2
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local = None
        grad_entree = self.save_X - self.D
        return grad_local,grad_entree
    
class Ilogit_and_KL(): 
    def __init__(self, Y) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.save_ilogit_z = None
        self.Y = Y
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def ilogit(self, z):
        z_exp = np.exp(z)
        sum_z_exp = np.sum(z_exp, axis=0)
        return z_exp / sum_z_exp
        
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_ilogit_z = self.ilogit(X)
        exp_x = np.exp(X)
        sum_exp = np.sum(exp_x, axis=0)
        scalar_prod = np.diag(np.dot(X.T, self.Y))
        return np.sum(np.log(sum_exp) - scalar_prod)

    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=self.save_ilogit_z - self.Y
        return grad_local,grad_entree
    
class Network() : 
    def __init__(self, list_layers) :
        self.list_layers = list_layers
        self.nb_params=sum([l.nb_params for l in list_layers]) # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        current_position = 0
        for l in self.list_layers:
            n = l.nb_params
            l.set_params(params[current_position:current_position+n])
            current_position += n
            
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        p = []
        for l in self.list_layers:
            new_p = l.get_params()
            if new_p is not None :
                p.append(new_p)
        return np.concatenate(p)
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Z = np.copy(X)
        for l in self.list_layers:
            Z = l.forward(Z)
        return Z
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        b = []
        gs = grad_sortie
        for l in reversed(self.list_layers):
            gl, ge = l.backward(gs)
            if gl is not None:
                b.append(gl)
            gs = ge
        b.reverse()
        grad_local = np.concatenate(b)
        grad_entree = ge
        return grad_local,grad_entree
    
