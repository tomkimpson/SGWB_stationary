
import pandas as pd 
import numpy as np 
from numpy import sin, cos
class PTA:
    """ 
    Define a PTA composed of 2 pulsars, and the obs plan
    """


    def __init__(self,Tobs,dt,seed=1):


        #Define some universal constants
        pc = 3e16     # parsec in m
        c  = 3e8      # speed of light in m/s

        #Load the pulsar data
        pulsars = pd.read_csv("../data/NANOGrav_pulsars.csv") #hardcoded
        Npsr = 2
        pulsars = pulsars.sample(Npsr,random_state=seed) # randomly select 2 pulsars  

       
        #Extract the parameters
        self.d_psr         = pulsars["DIST"].to_numpy()*1e3*pc/c         # this is in units of s^-1
        self.δ_psr         = pulsars["DECJD"].to_numpy()                 # radians
        self.α_psr         = pulsars["RAJD"].to_numpy()                  # radians
        
        #Some useful reshaping for vectorised calculations later
        self.d_psr          = self.d_psr.reshape(Npsr,1)
    
        #Pulsar positions as unit vectors
        self.q         = _unit_vector(np.pi/2.0 -self.δ_psr, self.α_psr) #shape(3,Npsr)


        #It is useful in calculate_gw to have the products q1*q1, q1*q2 precomputed
        self.q_products = np.zeros((Npsr,9))
        k = 0
        for n in range(Npsr):
            k = 0
            for i in range(3):
                for j in range(3):
                    self.q_products[n,k] = self.q[i,n]*self.q[j,n]
                    k+=1
        self.q_products = self.q_products.T # lazy transpose here to enable correct shapes for dot products later


        

        self.t = np.arange(0,Tobs,dt)
        self.Npsr = Npsr

"""
Given a latitude θ and a longitude φ, get the xyz unit vector which points in that direction 
"""
def _unit_vector(θ,φ):
    qx = sin(θ) * cos(φ)
    qy = sin(θ) * sin(φ)
    qz = cos(θ)
    return np.array([qx, qy, qz]) 
