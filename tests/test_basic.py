# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:24:21 2016

@author: marcobarsacchi

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
from mixtureDP import mixtureDP

__test__ = True
if __test__:
    # Try with generated dataset mixture of four binomials
    p1 = 0.3
    p2 = 0.7
    p3 = 0.45
    p4 = 0.9
    pVect = [p1, p2, p3, p4]
    nSample = 1000
    
    vSamples = []
    nTrial = 100.0
    for k in range(nSample):
        pVal = np.random.choice(pVect)
        vSamples.append(np.random.binomial(nTrial, pVal))
    trial = np.ones((1000))*nTrial
    success = np.array(vSamples)/nTrial
    
    
    myModel = mixtureDP.mixtureModelBin(success, trial, m=10, numIteration=300, pBeta=[1.,1.], pAlpha = [0.5,10.])
    c, bin_prob = myModel.run(samples= 1)

    
    plotting = True
    if plotting:
        import matplotlib.pyplot as plt
        print bin_prob
        plt.bar(np.unique(c),np.array([np.sum(c == c_k) for c_k in np.unique(c)]))
        plt.figure()
        plt.scatter(success,c,c=c)
