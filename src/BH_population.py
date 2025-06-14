import numpy as np
from scipy import stats 

class Universe:
    """ 
    Calculate the population of black holes which constitute the stochastic GW background.
    This involves randomly drawing 7 GW parameters (Ω,h,φ0,ψ,ι,α,δ) for M sources. 
    Uses NumPy/SciPy for fast, seedable random sampling from the required distributions.
    """


    def __init__(self,Ω_power_law_index,Ω_min,Ω_max,M,seed=None):

        #Assign arguments to class
        self.alpha = Ω_power_law_index
        self.Ω_min = Ω_min
        self.Ω_max = Ω_max
        self.M = M
        
        # Set random seed for reproducible results
        if seed is not None:
            np.random.seed(seed)

        # Sample GW parameters directly using NumPy/SciPy
        self.Ω = self._sample_power_law(self.alpha, self.Ω_min, self.Ω_max, M)
        self.h = np.full(M, 1.0)  # Delta function at 1.0
        self.φ0 = np.random.uniform(0.0, 2*np.pi, M)
        self.ψ = np.random.uniform(0.0, np.pi, M)
        self.ι = self._sample_sine(0.0, np.pi, M)
        self.δ = self._sample_cosine(-np.pi/2, np.pi/2, M)
        self.α = np.random.uniform(0.0, 2*np.pi, M)


    


    def _sample_power_law(self, alpha, minimum, maximum, size):
        """
        Sample from power law distribution: P(x) ∝ x^alpha
        Using inverse CDF method for exact sampling
        """
        if alpha == -1:
            # Special case: log-uniform distribution
            u = np.random.uniform(0, 1, size)
            return minimum * (maximum/minimum)**u
        else:
            # General power law case
            u = np.random.uniform(0, 1, size)
            a1 = alpha + 1
            return ((maximum**a1 - minimum**a1) * u + minimum**a1)**(1/a1)
    
    def _sample_sine(self, minimum, maximum, size):
        """
        Sample from sine distribution: P(x) ∝ sin(x)
        Using inverse CDF method
        """
        u = np.random.uniform(0, 1, size)
        # CDF of sin(x) from 0 to π is (1 - cos(x))/2
        # For general interval [a,b]: use transformation
        return np.arccos(np.cos(maximum) + u * (np.cos(minimum) - np.cos(maximum)))
    
    def _sample_cosine(self, minimum, maximum, size):
        """
        Sample from cosine distribution: P(x) ∝ cos(x)  
        Using inverse CDF method for interval [-π/2, π/2]
        """
        u = np.random.uniform(0, 1, size)
        # CDF of cos(x) from -π/2 to π/2 is (sin(x) + 1)/2
        return np.arcsin(2*u - 1)
    
    @classmethod
    def generate_batch(cls, Ω_power_law_index, Ω_min, Ω_max, M, num_realizations, seed=None):
        """
        Generate a batch of universe realizations for vectorized processing.
        
        Returns:
            Dictionary with batched parameters, each with shape (num_realizations, M)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate all realizations at once
        batch_Ω = np.array([cls._sample_power_law_static(Ω_power_law_index, Ω_min, Ω_max, M) 
                           for _ in range(num_realizations)])
        batch_h = np.ones((num_realizations, M))
        batch_φ0 = np.random.uniform(0.0, 2*np.pi, (num_realizations, M))
        batch_ψ = np.random.uniform(0.0, np.pi, (num_realizations, M))
        batch_ι = np.array([cls._sample_sine_static(0.0, np.pi, M) for _ in range(num_realizations)])
        batch_δ = np.array([cls._sample_cosine_static(-np.pi/2, np.pi/2, M) for _ in range(num_realizations)])
        batch_α = np.random.uniform(0.0, 2*np.pi, (num_realizations, M))
        
        return {
            'Ω': batch_Ω,
            'h': batch_h, 
            'φ0': batch_φ0,
            'ψ': batch_ψ,
            'ι': batch_ι,
            'δ': batch_δ,
            'α': batch_α
        }
    
    @staticmethod
    def _sample_power_law_static(alpha, minimum, maximum, size):
        """Static version of _sample_power_law for batch generation"""
        if alpha == -1:
            u = np.random.uniform(0, 1, size)
            return minimum * (maximum/minimum)**u
        else:
            u = np.random.uniform(0, 1, size)
            a1 = alpha + 1
            return ((maximum**a1 - minimum**a1) * u + minimum**a1)**(1/a1)
    
    @staticmethod
    def _sample_sine_static(minimum, maximum, size):
        """Static version of _sample_sine for batch generation"""
        u = np.random.uniform(0, 1, size)
        return np.arccos(np.cos(maximum) + u * (np.cos(minimum) - np.cos(maximum)))
    
    @staticmethod
    def _sample_cosine_static(minimum, maximum, size):
        """Static version of _sample_cosine for batch generation"""
        u = np.random.uniform(0, 1, size)
        return np.arcsin(2*u - 1)









        
