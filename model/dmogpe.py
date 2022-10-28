import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as rmse_score
from sklearn.cluster import KMeans

from tensorflow.random import set_seed

from scipy.spatial import distance
from sklearn.mixture import GaussianMixture as GMM

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from GPy.models import SparseGPRegression as SGPR
from GPy.models import GPRegression as GPR
from GPy.mappings.constant import Constant
from GPy.kern import RBF

from tqdm import tqdm

import seaborn as sns

import itertools
flatten = itertools.chain.from_iterable

# from autokeras import StructuredDataClassifier


class MetricsAndVisualisation:  
    def hard_prediction(self, X, return_variance=False):
        y_pred, y_pred_var = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        labs = np.argmax(self.dnn_model.predict(X),axis=-1)
        for lab in range(self.n_experts):
            x_ind = np.argwhere(labs == lab).flatten()
            buf_mu, buf_var = self.gp_experts[lab].predict(X[x_ind,:])
            y_pred[x_ind], y_pred_var[x_ind] = buf_mu.flatten(), buf_var.flatten()
        if return_variance:
            return y_pred, y_pred_var
        return y_pred
    
    def soft_prediction(self, X, return_variance=False):
        mu = np.zeros((X.shape[0], self.n_experts))
        var = np.zeros((X.shape[0], self.n_experts))        
        for i in range(self.n_experts):
            buf_mu, buf_var = self.gp_experts[i].predict(X)
            mu[:,i], var[:,i] = buf_mu.flatten(), buf_var.flatten()
        lab_probs = self.dnn_model.predict(X)
        y_pred = np.sum(lab_probs*mu,axis=1)
        y_pred_var = np.sum(lab_probs*(var + mu**2),axis=1) - y_pred**2
        if return_variance:
            return y_pred, y_pred_var   
        return y_pred
    
    def scoring(self, X_test, y_test):
        y_pred_hard = self.hard_prediction(X_test)
        y_pred_soft = self.soft_prediction(X_test)
        r2_hard_train = r2_score(y_test, y_pred_hard)
        r2_soft_train = r2_score(y_test, y_pred_soft)
        rmse_hard_train = rmse_score(y_test, y_pred_hard, squared=False)
        rmse_soft_train = rmse_score(y_test, y_pred_soft, squared=False)
        print(f"\nR^2. Hard_pred: {100*r2_hard_train:.2f} %. Soft_pred: {100*r2_soft_train:.2f} %.")
        print(f"RMSE. Hard_pred: {rmse_hard_train:.4f}. Soft_pred: {rmse_soft_train:.4f}.")
    
    def estimate_density_hard(self, X, N=10000, y=None):
        lab = np.argmax(self.dnn_model.predict(X.reshape(1,-1)),axis=-1)[0]
        mu = self.gp_experts[lab].predict(X.reshape(-1,1).T)[0].flatten()
        var = self.gp_experts[lab].predict(X.reshape(-1,1).T)[1].flatten()
        samples = np.random.normal(loc=mu,scale=np.sqrt(var),size=N)
        plt.figure(figsize=(8,6))
        sns.distplot(samples,norm_hist=True,axlabel=r"$\hat{y}$")
        plt.axvline(mu,label="Hard Prediction, $y^* = {}$".format(np.round(mu[0],3)),color="green")
        if y is not None:
            plt.axvline(y,label="True, $y = {}$".format(np.round(y[0],3)),color="red")
        plt.title("Hard density estimation of the prediction at $x^*={}$".format(np.round(X[0],3)))
        plt.legend() 
    
    def estimate_density_soft(self, X, N=10000, y=None):
        lab = self.dnn_model.predict(X.reshape(1,-1))[0]
        mu = np.array([self.gp_experts[i].predict(X.reshape(-1,1).T)[0].flatten() for i in range(len(lab))]).flatten()
        var = np.array([self.gp_experts[i].predict(X.reshape(-1,1).T)[1].flatten() for i in range(len(lab))]).flatten()
        samples = []
        for i in range(len(mu)):
            samples.append(np.random.normal(loc=mu[i],scale=np.sqrt(var[i]),size=int(lab[i]*N)).flatten())
        samples = np.array(list(flatten(samples)))
        plt.figure(figsize=(8,6))
        sns.distplot(samples,norm_hist=True,axlabel=r"$\hat{y}$")
        plt.axvline(np.sum(lab*mu),label="Soft Prediction, $y^* = {}$".format(np.round(np.sum(lab*mu),3)),color="green")
        if y is not None:
            plt.axvline(y,label="True, $y = {}$".format(np.round(y[0],3)),color="red")
        plt.title("Soft density estimation of the prediction at $x^*={}$".format(np.round(X[0],3)))
        plt.legend()
        plt.show()
        
    def plot(self, X, y, X_test, y_test):
        if self.dim == 1:
            y_pred_soft, y_pred_var = self.soft_prediction(X, return_variance=True)
            y_pred_hard  = self.hard_prediction(X)
            x_sorted = np.argsort(X.flatten())

            ci = 2*np.sqrt(y_pred_var)
            fig, ax = plt.subplots(figsize=(12,8))
            ax.scatter(X,y,label="True Data",alpha=0.6,color='darksalmon')
            ax.plot(X[x_sorted],y_pred_hard[x_sorted],label='Hard Predictions')
            ax.plot(X[x_sorted],y_pred_soft[x_sorted],label='Soft Predictions')
            ax.fill_between(X[x_sorted].flatten(), (y_pred_soft[x_sorted]-ci), (y_pred_soft[x_sorted]+ci), color='b', alpha=.1, label="Soft $2\sigma$ Credible Interval")
            ax.set_xlabel("X")
            ax.set_ylabel("y")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.title("Test data: $R^2_{hard} = $"+"{}%,".format(np.round(100*r2_score(y_test,self.hard_prediction(X_test)),2)) +
                     "  $R^2_{soft} = $"+"{}%.".format(np.round(100*r2_score(y_test,self.soft_prediction(X_test)),2)))
            plt.show()  
          
        
        
class DeepMixtureGPE(MetricsAndVisualisation):  
    """
    Main class
    """   
    def __init__(self, n_experts=3, sparse_gp=False, fix_ind_pnts=True, initial_clustering='GMM', random_state=None):
        self.dim = None        
        self.dnn_model = None
        self.gp_experts = None
        self.z = None
        self.log_poster = None
        self._automl = False
        
        self.n_experts = n_experts
        self._rnd_state = random_state
        self._sparse_gp = sparse_gp
        self._fix_ind_pnts = fix_ind_pnts
        self._initial_clustering = initial_clustering
     
    @staticmethod
    def estimate_n_experts(data, max_K=11, n_init=30, max_iter=300, criterion='elbow', random_state=None):     
        """
        Estimate number of experts.
        (i) Visually, using the Elbow method
        (ii) Analitically, using BIC or AIC criteria

        Data should be of the same scale. 
        """
        def compute_bic(kmeans,X):
            # assign centers and labels
            centers = [kmeans.cluster_centers_]
            labels  = kmeans.labels_
            #number of clusters
            m = kmeans.n_clusters
            # size of the clusters
            n = np.bincount(labels)
            #size of data set
            N, d = X.shape
            #compute variance for all clusters beforehand
            cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
                     'euclidean')**2) for i in range(m)]) 
            const_term = 0.5 * m * np.log(N) * (d+1) 

            BIC = np.sum([n[i] * np.log(n[i]) -
                       n[i] * np.log(N) -
                     ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                     ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
            return BIC

        def compute_aic(k,X,n_init=30,random_state=1):
            return GMM(n_components=k, n_init=n_init,random_state=random_state).fit(X).aic(X)

        distortions = []
        aic_vals = []
        bic_vals = []
        for k in range(1,max_K+1):
            if criterion=='AIC':
                aic_vals.append(compute_aic(k,data,n_init=n_init,random_state=random_state))
                continue
            kmeanModel = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state).fit(data)
            if criterion=='BIC':
                bic_vals.append(compute_bic(kmeanModel,data))
            if criterion=='elbow':    
                distortions.append(kmeanModel.inertia_)    

        plt.figure(figsize=(14,8))
        if criterion=='AIC':
            print(f"Number of clusters is {np.argmin(aic_vals)+1}")
            plt.plot(range(1,max_K+1), aic_vals, 'bx-')
            plt.ylabel('AIC value')
        if criterion=='BIC':
            print(f"Number of clusters is {np.argmax(bic_vals)+1}")
            plt.plot(range(1,max_K+1), bic_vals, 'bx-')
            plt.ylabel('BIC value')  
        if criterion=='elbow': 
            plt.plot(range(1,max_K+1), distortions, 'bx-')
            plt.ylabel('Distortion')
        plt.xlabel('k, num of clusters')
        plt.grid(alpha=0.6)
        plt.show()
        
    def __initial_clustering_step(self, X, y, random=False):
        if random:
            np.random.seed(self._rnd_state)
            self.z = np.random.randint(low=0, high=self.n_experts, size=(y.shape[0]))
        else: 
            if self._initial_clustering != 'GMM':
                kmeanModel = KMeans(n_clusters=self.n_experts, n_init=10, max_iter=100, random_state=self._rnd_state)
                kmeanModel.fit(np.concatenate((X,self.dim*y), axis=1))
                self.z = kmeanModel.labels_   
            else:
                gmm = GMM(n_components=self.n_experts, n_init=10, max_iter=100, covariance_type='full', random_state=self._rnd_state, verbose=0)
                self.z = gmm.fit_predict(np.concatenate((X,self.dim*y), axis=1))          
        
    def __clustering_step(self, X, y):
        gp_means = np.zeros((X.shape[0], self.n_experts))
        gp_variances = np.zeros((X.shape[0], self.n_experts))
        log_probs = -np.log(self.dnn_model.predict(X))
        for k in range(self.n_experts):
            buf_means, buf_variances = self.gp_experts[k].predict(X)
            gp_means[:,k], gp_variances[:,k] = buf_means.flatten(), buf_variances.flatten()
        s = log_probs + 0.5*np.log(2*np.pi*gp_variances) + 0.5*(y-gp_means)**2/gp_variances 
        self.z = np.argmin(s, axis=1)
        
    def __gating_network_step(self, X, verbose):
        set_seed(self._rnd_state)
        np.random.seed(self._rnd_state)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=30, restore_best_weights=False)
        
        # DNN Model 
        self.dnn_model = Sequential()
        l2_fac = 0.0001
        self.dnn_model.add(Dense(200, input_dim=self.dim, kernel_regularizer=l2(l2_fac), bias_regularizer=l2(l2_fac), activation='relu'))
        self.dnn_model.add(Dense(40, kernel_regularizer=l2(l2_fac), bias_regularizer=l2(l2_fac), activation='relu'))
        self.dnn_model.add(Dense(30, kernel_regularizer=l2(l2_fac), bias_regularizer=l2(l2_fac), activation='relu'))
        self.dnn_model.add(Dense(self.n_experts, kernel_regularizer=l2(l2_fac), bias_regularizer=l2(l2_fac), activation='softmax'))
        self.dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        dummy_z = np_utils.to_categorical(LabelEncoder().fit_transform(self.z))
        
        self.dnn_model.fit(X, dummy_z, batch_size=16, epochs=500, verbose=verbose, validation_split=0.1, callbacks=[es])
        if verbose:
            print(f"DNN train accuracy: {self.dnn_model.evaluate(X, dummy_z)[1]:.3f}")
        
        # uncomment below for Auto Keras architecture tuning
#         dummy_z = np_utils.to_categorical(encoded_z)
#         if self._automl:
#             search = StructuredDataClassifier(max_trials=5)
#             search.fit(x=X, y=dummy_z, epochs=8, verbose=0, validation_split=0.1)    
#             self.dnn_model = search.export_model()
#             self.dnn_model.fit(X_train, dummy_z, batch_size=8, epochs=500, callbacks=[es],validation_split=0.1, verbose=0)
#             self._automl = False
#         else:
#             self.dnn_model.fit(X_train, dummy_z, batch_size=8, epochs=500, callbacks=[es],validation_split=0.1, verbose=0)
    
    
    def __experts_step(self, X, y, verbose):
        np.random.seed(self._rnd_state)
        classes = []
        for j in range(self.n_experts):
            classes.append(np.argwhere(self.z == j).flatten())
            
        self.gp_experts = []
        for k in range(self.n_experts):
            N = X[classes[k]].shape[0]
            if N == 0:
                print(f"No points were assigned to expert {k}")
                continue
            kernel = RBF(input_dim=self.dim, ARD=False)
            mf = Constant(self.dim,1,np.mean(y[classes[k]]))           
            if self._sparse_gp:
                if self.dim < 5:
                    M = round(np.log(N)**self.dim)
                else:
                    M = min(round(0.1*N), round(np.log(N)**self.dim))
                ind_clstrs = KMeans(n_clusters=M, random_state=self._rnd_state).fit(X[classes[k]])
                ind_pnts = ind_clstrs.cluster_centers_
                self.gp_experts.append(SGPR(X[classes[k]],y[classes[k]],kernel=kernel,
                                            mean_function=mf,Z=ind_pnts,normalizer=True))            
                if self._fix_ind_pnts:
                    self.gp_experts[-1].inducing_inputs.fix()  
            else:
                self.gp_experts.append(GPR(X[classes[k]],y[classes[k]],kernel=kernel,mean_function=mf,normalizer=True))           
            self.gp_experts[-1].optimize_restarts(num_restarts=2, messages=bool(verbose)) # – for several restarts with random params
            
    def fit(self, X, y, n_iterations=6, mode='CCR', verbose=0):
        """
        Main algorithm
        """
        assert mode in ['CCR', 'CCR-MM', 'MM2r'], "mode should take 'CCR', 'CCR-MM', or 'MM2r'"
   
        self.log_poster = []
        
        self.dim = X.shape[1]
        if mode == 'MM2r':
            random_allocation = True
            n_iterations = 2
        else:
            random_allocation = False
        
        self.__initial_clustering_step(X, y, random=random_allocation)  # Clustering
        
        if mode == 'CCR':
            self.__gating_network_step(X, verbose)  # DNN training
            log_probs = -np.log(self.dnn_model.predict(X))  # Additional clustering step for CCR
            self.z = np.argmin(log_probs, axis=1)
            self.__experts_step(X, y, verbose)  # Tuning of experts
        else:
            for i in tqdm(range(n_iterations)):
                if i != 0:
                    self.__clustering_step(X, y)
                self.__experts_step(X, y, verbose)  # Tuning of experts
                self.__gating_network_step(X, verbose)  # DNN training
                
                # control of NLL of posterior
                gp_means = np.zeros((X.shape[0],1))
                gp_variances = np.zeros((X.shape[0],1))
                log_probs = -np.log(np.max(self.dnn_model.predict(X),axis=1))
                for z_i, z in enumerate(self.z):
                    buf_means, buf_variances = self.gp_experts[z].predict(X[z_i].reshape(1,-1))
                    gp_means[z_i], gp_variances[z_i] = buf_means.flatten(), buf_variances.flatten()
                gp_means, gp_variances = gp_means.flatten(), gp_variances.flatten()
                s = log_probs + 0.5*np.log(2*np.pi*gp_variances) + 0.5*(y-gp_means)**2/gp_variances
                self.log_poster.append(np.sum(np.min(s, axis=1)))  
                
                if i != 0 and np.abs(self.log_poster[-1]-self.log_poster[-2]) <= 1e-6:
                    break