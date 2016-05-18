# Description
Class for performing mixture model over a set of successes. 
The algorithm implements a mixture of binomial distribution, with Dirichlet
Process prior having beta distribution as base distribution. Furthermore a
beta distribution can be selected as prior over the parameter alpha.
Due to auxiliary variable structure of the algorithm non conjugate priors
can be used.
