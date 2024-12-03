import bilby 
import numpy as np 

class BlackHolePopulation:
    """ 
    Calculate the population of black holes which constitute the stochastic GW background.
    This involves randomly drawing 7 GW parameters (Ω,h,φ0,ψ,ι,α,δ) for M sources. 
    We use the bilby package to do do the random sampling; note that bilby does not currently let a user seed the sampling process
    Se e.g. https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/prior/analytical.py
    """


    def __init__(self,Ω_power_law_index,Ω_min,Ω_max):

        self.alpha = Ω_power_law_index
        self.Ω_min = Ω_min
        self.Ω_max = Ω_max


    def _gw_priors(self):
        """
        Define the priors on the 7 GW parameters.
        Pass 3 arguments: Ω_power_law_index,Ω_min,Ω_max which define the power law prior on Ω
        Note that h has a unit delta function prior - all sources have the same unit amplitude
        """


        priors = bilby.core.prior.PriorDict()
        priors['Ω']  = bilby.core.prior.PowerLaw(alpha=self.alpha,minimum=self.Ω_min,maximum=self.Ω_max)
        priors['h']  = bilby.core.prior.DeltaFunction(1.0)
        priors['φ0'] = bilby.core.prior.Uniform(0.0, 2*np.pi)
        priors['ψ']  = bilby.core.prior.Uniform(0.0, np.pi)
        priors['ι']  = bilby.core.prior.Sine(0.0, np.pi)
        priors['δ']  = bilby.core.prior.Cosine(-np.pi/2, np.pi/2)
        priors['α']  = bilby.core.prior.Uniform(0.0, 2*np.pi)

        return priors





class Universe(BlackHolePopulation):
    def __init__(self,M):

        priors  = self._gw_priors()
        samples = priors.sample(M)

        #Manually extract from the dictionary and make them attributes of the class - easier to handle later
        self.Ω = samples['Ω']
        self.h = samples['h']
        self.φ0 = samples['φ0']
        self.ψ = samples['ψ']
        self.ι = samples['ι']
        self.δ = samples['δ']
        self.α = samples['α']


        self.M = M








        
