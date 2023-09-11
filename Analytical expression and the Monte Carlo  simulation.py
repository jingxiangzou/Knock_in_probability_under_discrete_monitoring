import numpy as np
import pandas as pd
import random
import scipy as sp
from scipy.special import zeta
import matplotlib.pyplot as plt
import time


class testclass:
    def __init__(self, sigma, duration, instants, s0, s1, ki):
        """
        :param sigma: volatility of stock return 
        :param duration: the whole period, in units of year(s)
        :param instants: number of monitoring instants
        :param s0: starting price
        :param s1: terminal price
        :param ki: knock-in price 
        """
        self.sig = sigma
        self.T = duration
        self.m = instants
        self.s0 = s0
        self.s1 = s1
        self.ki = ki
        
        
    def ana_appro(self):
        """
        analytical approximation as per the hypothesis.
        """
        # the beta coefficient as per the paper 
        beta = -1 * zeta(1/2) / np.sqrt(2 * np.pi)
        
        # the knock-in barrier with continuity adjustment 
        adj_ki = self.ki * np.exp(-1 * beta * self.sig 
                                  * np.sqrt(self.T/self.m))
        
        return np.exp((-2.0)/ (self.T * self.sig * self.sig) 
                      * np.log(adj_ki / self.s0) * np.log(adj_ki / self.s1))
    
    
    def ana_cont(self):
        """
        the knock-in probability when monitoring is continuous.
        """
        return np.exp(-2 * np.log(self.ki/self.s1) * 
                      np.log(self.ki/self.s0) / (self.T * self.sig * self.sig))
    
        
    def mot_sim(self, nsim):
        """
        Monte Carlo Simulation
        """
        a = np.random.normal(0, self.sig * np.sqrt(self.T / self.m), self.m * nsim)
        b = a.reshape((nsim, self.m))
        
        # sum over columns for each row
        c = np.cumsum(b, axis=1) 

        # the following translates brownian motion into brownian bridge
        vmt = np.eye(self.m)
        ary = np.zeros(self.m)
        ary[:-1] = -1/self.m * np.arange(1, self.m)
        vmt[-1, :] = ary
        mat = c @ vmt

        d = np.array(list(np.log(self.s1 / self.s0) / self.m * np.arange(1, self.m + 1)) * nsim)
        e = d.reshape((nsim, self.m))
        nmat = np.add(mat, e)
        
        # it is consider a 'knock-in' when price dynamics goes beneath ki
        return sum(nmat.min(axis=1) < np.log(self.ki / self.s0)) / nsim


if __name__ == '__main__':
    test1 = testclass(0.3, 1/12, 21, 100, 95, 90)
    print(test1.ana_appro())
    start = time.time()
    print(test1.mot_sim(5000000)) # 500万
    end = time.time()
    
    # the time it takes to work out one probability with simulation
    # we can see that the two results are considerably close 
    print(end - start) 

    df =pd.DataFrame(columns=['instants', 'sigma', 's1', 'ana_prob', 
                              'mc_prob', 'discrepancy', 'continuous', 
                              'ratio'])
    tme = 0
    for sig in np.arange(0.15, 0.45, 0.02):
        for s in np.arange(91, 96, 0.1):
            for m in np.array([5, 12, 21, 50]):
                tc = testclass(sig, 1/12, m, 100, s, 90)
                ana_p = tc.ana_appro()
                mc_p = tc.mot_sim(5000000)
                tme += 1
                # the iteration indicator is an integar 
                # a total of 3200 iterations will be excuted 
                # which takes about 3 hours
                print(tme)
                new_row = pd.DataFrame({'instants': m, 'sigma': sig, 's1': s, 
                                        'ana_prob': ana_p, 'mc_prob': mc_p,
                                        'discrepancy': abs(ana_p - mc_p), 
                                        'continuous': tc.ana_cont(), 
                                        'ratio':abs(ana_p - mc_p) / ana_p},
                                        index=[0])
                df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)
    
    # this excel file is then used for data analysis 
    # which will be done using the 'data analysis.py' file
    df.to_excel('sresinf.xlsx')
    



