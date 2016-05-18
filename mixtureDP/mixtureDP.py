# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:24:21 2016

@author: marcobarsacchi
Class for performing mixture model over a set of successes. 
The algorithm implements a mixture of binomial distribution, with Dirichlet
Process prior having beta distribution as base distribution. Furthermore a
beta distribution can be selected as prior over the parameter alpha.
Due to auxiliary variable structure of the algorithm non conjugate priors
can be used.

This file is part of MixtureDP.

MixtureDP is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MixtureDP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MixtureDP.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.special
import logging
import time
import pickle

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Mixture')


class mixtureModelBin(object):
    """ Class for modelling success vector as mixture of binomial distribution.
    
    Dirichlet Process prior (:math:`DP( \\alpha, G_0)`) is used to avoid 
    specification of the number of clusters. Beta distribution is used as 
    base measure for the DP. Implementation follows Algorithm 8 from [1]_.
    Moreover sampling for alpha, according to its prior is implemented via 
    auxiliary variable, as suggested in [2]_.
    
    Parameters
    ----------
    success: array of float
        vector containing success values
    trial: array of int
        vector containing number of trial for each datum
    alpha: float, optional (default: 0.1)
        alpha parameter for the Dirichlet Process
    numIteration: int, optional (default: 200)
        number of iteration for the chain
    pBeta: [float, float], optional (default: [1.0, 1.0])
        parameters for the base beta distribution
    pAlpha: [float, float], optional
        parameters for the gamma prior over alpha; values of [1.0, 4.]
        might be a good starting point.
    m: int, optional (default: 10)
        number of auxiliary variables to be used in gibbs sampling
    
    Attributes
    ----------
    samples: list of samples
        Each sample is made up of a vector of cluster indicators, for each 
        datum, and a vector of parameters.
    
    Methods
    -------
    run():
        run the chain
        
    map_estimation():
        perform maximum a posteriori estimation
        
    References
    ----------
        
    .. [1] Radford M. Neal "Markov Chain Sampling Methods for Dirichlet Process 
        Mixture Models" Journal of Computational and Graphical Statistics
        Vol. 9, No. 2 (Jun., 2000), pp. 249-265.
    .. [2] Michael D. Escobar "Bayesian Density Estimation and Inference Using
        Mixtures" Journal of the American Statistical Association
        Volume 90, Issue 430, 1995, pp. 577-588.
        
    """
    def __init__(self, success, trial, alpha=0.1, m=10, numIteration=200, pBeta=[2.0,1.0], pAlpha=None, pickledSample=None):
        # Some definitions
    
        self.success =  success
        self.trial = trial
        self.numIteration = numIteration
        self.alpha = alpha
        self.n = success.shape[0]
        self.a = pBeta[0]
        self.b = pBeta[1]
        self.alphaPrior = False
        self.m = m
        self.samples = []
        self.pickledSample = pickledSample
        
        # A prior over alpha is defined
        if pAlpha:            
            self.alphaPrior = True
            self.a_alpha = pAlpha[0]
            self.b_alpha = pAlpha[1]
    
    def run(self, samples=1):
        """Run the gibbs sampling.
        
        Parameters
        ----------
        samples: int, optional (default=1)
            number of samples to be given back
            
        Return
        ------
        c: array
            cluster labels from the data
        bin_prob: array
            cluster parameters. 
        
        """
        # Initialization
        self.c = np.zeros(self.success.shape, dtype='int')
        self.bin_prob = np.array([0.5])
        
        # Start Looping
        timeStart = time.time()
        for k in range(self.numIteration+samples):
            logging.info("Doing Iteration %d of %d. Actual k: %d" % (k+1,self.numIteration+samples, len(self.bin_prob)))
            
            # Iterating on the data 
            for i in range(self.n):
                # Search for singletons
                relabel = None
                if np.sum(self.c == self.c[i]) == 1:
                    # Remove singletons
                    relabel = self.bin_prob[self.c[i]]
                    # De-assign freq_value for the cluster
                    self.bin_prob = np.concatenate((self.bin_prob[0:self.c[i]],self.bin_prob[self.c[i]+1:len(self.bin_prob)]))
                    # Renumber clusters
                    self.c[self.c>self.c[i]]=self.c[self.c>self.c[i]]-1
                    self.c[i] = 0                    
                    #self.c[i] = np.unique(self.c[i]) 
                
                
                # Sample new indicators
                # probability for one of the current cluster
                newDraw = self._sampleFromBase(self.m, i)
                
                # If relabelling has occurred the first new sample
                # is not really new at all
                if relabel:
                    newDraw[0] = relabel
                param_aux = np.concatenate((self.bin_prob, newDraw))
                p_t = self._currentProb(self.success, self.trial, self.c, param_aux, self.alpha, i, self.n, self.m)
                
                # Full probability vector, normalized    
                p_t = p_t /np.sum(p_t)
                
                sampled = np.random.choice(np.arange(0,len(np.unique(self.c))+self.m,dtype='int'), size=1,p=p_t)
                if sampled >= len(np.unique(self.c)):
                    newC = True
                    self.c[i] = len(np.unique(self.c))
                else:
                    newC = False
                    self.c[i] = sampled
                
                # If newC sample parameter for cluster c
                if newC:
                    newP =  param_aux[sampled]
                    self.bin_prob = np.concatenate((self.bin_prob, np.array(newP)))
            
                  
            # Now start updating parameters
            for i in range(len(np.unique(self.c))):
                # Number of data points in the i cluster
                num_i = np.sum(self.c == i)
                
                # Update make sense if the cluster containts at least two elements
                if num_i > 1:
                    np_ind = np.where(self.c==i)
                    tempk = self.success[np_ind]*self.trial[np_ind]
                    tempn = self.trial[np_ind]
                    num_val = len(tempn)
                    k_like = np.sum(tempk)
                    n_like = np.sum(tempn)
                    a_beta = k_like + self.a*num_val -num_val + 1
                    b_beta = n_like+self.b*num_val - k_like -num_val + 1
                    self.bin_prob[i] = np.random.beta(a_beta, b_beta)
            
            
            if self.alphaPrior:
                # number of actual clusters
                k_clust = len(self.bin_prob)
                
                # Sample eta and alpha
                # eta is a beta distributed random variable
                eta = np.random.beta(self.alpha+1,self.n)
                # 
                pn = 1.0 / (1.0+(self.n*(self.b_alpha - np.log(eta)))/(self.a_alpha+ k_clust-1))
                unif = np.random.uniform()
                if unif<pn:
                    self.alpha = self._gamma_rvs(self.a_alpha+k_clust, self.b_alpha - np.log(eta))
                else:
                    self.alpha = self._gamma_rvs(self.a_alpha+k_clust-1, self.b_alpha - np.log(eta))
            
            # Save samples
            # Item must be copied, otherwise the will change each iteration
            if k >= self.numIteration:
                self.samples.append([self.c.copy(), self.bin_prob.copy()])                
        logging.info("Simulation terminated in %f seconds." % (time.time()-timeStart))
        logging.info("%d clusters have been spotted." % len(self.bin_prob))
        
        if self.pickledSample:
            logging.info("Saving sample file to %s ..." % self.pickledSample)
            try:
                pickle.dump(self.samples, self.pickledSample)
            except Exception as e:
                logging.warn("Error writing file: %s" % e.message)
            
        if self.alphaPrior:
            logging.info("Final sampled alpha: %f ." % self.alpha)
        return self.c, self.bin_prob
    
        
    # Functions 
    def _currentProb(self, success, rD, c, bin_prob, alpha, i, n, m):
        # Estimate probability for the current clusters    
        c1 = np.concatenate((c[0:i],c[i+1:]))        
        cProbvect = np.zeros(len(bin_prob))
        # Given the algorithm used here, a simple form from c == c_i / (N + alpha - 1) F(x_i, theta_j)
        for j in range(len(np.unique(c1))):
            cProbvect[j] = len(np.where(c1==np.unique(c1)[j])[0])/float(n-1+alpha) * self._binomialEval(bin_prob[j],success[i]*rD[i],rD[i])   
        for j in range(len(np.unique(c1)),len(np.unique(c1))+m):
            cProbvect[j] = (alpha/m)/float(n-1+alpha) * self._binomialEval(bin_prob[j],success[i]*rD[i],rD[i])
        return cProbvect
    
    
    def _sampleFromBase(self, m, i):
        # Sample from base distribution, beta in this case
        sampledBase = np.random.beta(self.a, self.b, size=m)
        # Or maybe, 
        #n_b = self.trial[i]
        #k_b = self.success[i] * self.trial[i]
        #sampledBase = np.random.beta(k_b+self.a, n_b+self.b-k_b, size=m)
        return sampledBase
        
    
    def _binomialEval(self, p, k, n):
        # Evaluate probability for binomial distribution
        bino = scipy.special.binom(n,k) * p**k * (1-p)**(n-k)
        return bino
    
    
    def _sampleParameters(self, n_p, k_p):
        # In this case we are sampling from beta
        # the product of binom(k| p;n) * beta(p;a,b) = beta(p|k;a,b) 
        # i.e. Beta(k+a ,n+b-k)
    
        return np.random.beta(k_p+self.a, n_p+self.b-k_p)
    
    
    def _gamma_rvs(self,a, b):
        """ Random variable with gamma distribution.
        
        """
    
        sample_gamma = np.random.gamma(a,1./b)
        return sample_gamma
    

        