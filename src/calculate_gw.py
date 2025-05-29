


import numpy as np 
import jax.numpy as jnp
from jax import jit, vmap
import jax
jax.config.update("jax_enable_x64", True)

from numpy import sin, cos


class GW:
    """ 
    For a population of M black holes, calculate the timeseries a(t)
    """

    def __init__(self,universe_i,PTA):

        #Gw parameters
        self.Ω=universe_i.Ω
        self.δ = universe_i.δ
        self.α = universe_i.α
        self.ψ = universe_i.ψ
        self.h=universe_i.h
        self.ι = universe_i.ι
        self.φ0 = universe_i.φ0
  
        #PSR related quantities
        self.q = PTA.q.T
        self.q_products = PTA.q_products
        self.t = PTA.t
        self.d_psr=PTA.d_psr

        #Shapes
        self.M,self.T,self.N = universe_i.M,len(PTA.t),PTA.Npsr 

    """ 
    Sole function of the class
    """
    def compute_a(self):

        m,n                 = _principal_axes(np.pi/2.0 - self.δ,self.α,self.ψ) # Get the principal axes. declination converted to a latitude 0-π. Shape (K,3)
        gw_direction        = np.cross(m,n)                                     # The direction of each source. Shape (,3)
        e_plus,e_cross      = polarisation_tensors(m.T,n.T)                     # The polarization tensors. Shape (3,3,K)
        hp,hx               = h_amplitudes(self.h,self.ι)                       # The plus and cross amplitudes. Can also do h_amplitudes(h*Ω**(2/3),ι) to add a frequency dependence
        
        dot_product         = 1.0 + self.q @ gw_direction.T                     # Shape (N,M)


        #Amplitudes
        Hij_plus             = (hp * e_plus).reshape(9,self.M).T # shape (3,3,M) ---> (9,M)---> (M,9). Makes it easier to later compute the sum q^i q^j H_ij
        Hij_cross            = (hx * e_cross).reshape(9,self.M).T 

        Fplus = np.dot(Hij_plus,self.q_products) #(M,Npsr)
        Fcross = np.dot(Hij_cross,self.q_products) #(M,Npsr)


        #Phases
        earth_term_phase  = np.outer(self.Ω,self.t).T + self.φ0 # Shape(T,M)
        phase_correction  =  self.Ω*dot_product*self.d_psr
        pulsar_term_phase = earth_term_phase.T.reshape(self.M,self.T,1) +phase_correction.T.reshape(self.M,1,self.N) # Shape(M,T,N)


        #Trig terms
        cosine_terms = cos(earth_term_phase).reshape(self.T,self.M,1) - cos(pulsar_term_phase).transpose(1, 0, 2)
        sine_terms   = sin(earth_term_phase).reshape(self.T,self.M,1) - sin(pulsar_term_phase).transpose(1, 0, 2)


        #Redshift per pulsar per source over time
        zplus  = Fplus*cosine_terms
        zcross = Fcross*sine_terms
        z = (zplus+zcross)/(2*dot_product.T) # (T,M,N)

        #Put it all together
        a = np.sum(z,axis=1) #the GW on the nth pulsar at time t is the sum over the M GW sources. Shape (T,Npsr)
        

        return a

    def compute_a_jax(self):
        """
        JAX-optimized version of compute_a for maximum performance
        """
        # Convert arrays to JAX arrays
        δ_jax = jnp.array(self.δ)
        α_jax = jnp.array(self.α) 
        ψ_jax = jnp.array(self.ψ)
        h_jax = jnp.array(self.h)
        ι_jax = jnp.array(self.ι)
        Ω_jax = jnp.array(self.Ω)
        φ0_jax = jnp.array(self.φ0)
        q_jax = jnp.array(self.q)
        q_products_jax = jnp.array(self.q_products)
        t_jax = jnp.array(self.t)
        d_psr_jax = jnp.array(self.d_psr)
        
        return _compute_a_jax_compiled(
            δ_jax, α_jax, ψ_jax, h_jax, ι_jax, Ω_jax, φ0_jax,
            q_jax, q_products_jax, t_jax, d_psr_jax
        )
    
    @classmethod
    def compute_a_batch(cls, batch_universe_params, PTA):
        """
        Vectorized computation of a(t) for multiple universe realizations.
        
        Args:
            batch_universe_params: Dict with keys ['Ω', 'δ', 'α', 'ψ', 'h', 'ι', 'φ0']
                                 Each value has shape (num_realizations, M)
            PTA: PTA instance with pulsar parameters
        
        Returns:
            JAX array with shape (num_realizations, T, N)
        """
        # Convert to JAX arrays
        batch_δ = jnp.array(batch_universe_params['δ'])
        batch_α = jnp.array(batch_universe_params['α']) 
        batch_ψ = jnp.array(batch_universe_params['ψ'])
        batch_h = jnp.array(batch_universe_params['h'])
        batch_ι = jnp.array(batch_universe_params['ι'])
        batch_Ω = jnp.array(batch_universe_params['Ω'])
        batch_φ0 = jnp.array(batch_universe_params['φ0'])
        
        # PTA parameters (same for all realizations)
        q_jax = jnp.array(PTA.q.T)
        q_products_jax = jnp.array(PTA.q_products)
        t_jax = jnp.array(PTA.t)
        d_psr_jax = jnp.array(PTA.d_psr)
        
        return _compute_a_batch_jax_compiled(
            batch_δ, batch_α, batch_ψ, batch_h, batch_ι, batch_Ω, batch_φ0,
            q_jax, q_products_jax, t_jax, d_psr_jax
        )



#These are all pure functions which exist outside of the class
def _principal_axes(θ,φ,ψ):


    m = np.zeros((len(θ),3)) #size M GW sources x 3 component directions    
    m[:,0] = sin(φ)*cos(ψ) - sin(ψ)*cos(φ)*cos(θ)
    m[:,1] = -(cos(φ)*cos(ψ) + sin(ψ)*sin(φ)*cos(θ))
    m[:,2] = sin(ψ)*sin(θ)


    n = np.zeros_like(m)
    n[:,0] = -sin(φ)*sin(ψ) - cos(ψ)*cos(φ)*cos(θ)
    n[:,1] = cos(φ)*sin(ψ) - cos(ψ)*sin(φ)*cos(θ)
    n[:,2] = cos(ψ)*sin(θ)


    return m,n



def polarisation_tensors(m, n):
    x, y = m.shape

    #See e.g. https://stackoverflow.com/questions/77319805/vectorization-of-complicated-matrix-calculation-in-python
    ma = m.reshape(x, 1, y)
    mb = m.reshape(1, x, y)

    na = n.reshape(x, 1, y)
    nb = n.reshape(1, x, y)

    e_plus = ma*mb -na*nb
    e_cross = ma*nb +na*mb

    return e_plus,e_cross



"""
Get the hplus and hcross amplitudes
"""
def h_amplitudes(h,ι): 
    return h*(1.0 + cos(ι)**2),h*(-2.0*cos(ι)) #hplus,hcross


# JAX-optimized functions
@jit
def _principal_axes_jax(θ, φ, ψ):
    """JAX version of _principal_axes"""
    m = jnp.zeros((len(θ), 3))
    m = m.at[:, 0].set(jnp.sin(φ) * jnp.cos(ψ) - jnp.sin(ψ) * jnp.cos(φ) * jnp.cos(θ))
    m = m.at[:, 1].set(-(jnp.cos(φ) * jnp.cos(ψ) + jnp.sin(ψ) * jnp.sin(φ) * jnp.cos(θ)))
    m = m.at[:, 2].set(jnp.sin(ψ) * jnp.sin(θ))
    
    n = jnp.zeros_like(m)
    n = n.at[:, 0].set(-jnp.sin(φ) * jnp.sin(ψ) - jnp.cos(ψ) * jnp.cos(φ) * jnp.cos(θ))
    n = n.at[:, 1].set(jnp.cos(φ) * jnp.sin(ψ) - jnp.cos(ψ) * jnp.sin(φ) * jnp.cos(θ))
    n = n.at[:, 2].set(jnp.cos(ψ) * jnp.sin(θ))
    
    return m, n

@jit
def _polarisation_tensors_jax(m, n):
    """JAX version of polarisation_tensors"""
    x, y = m.shape
    ma = m.reshape(x, 1, y)
    mb = m.reshape(1, x, y)
    na = n.reshape(x, 1, y)
    nb = n.reshape(1, x, y)
    
    e_plus = ma * mb - na * nb
    e_cross = ma * nb + na * mb
    
    return e_plus, e_cross

@jit
def _h_amplitudes_jax(h, ι):
    """JAX version of h_amplitudes"""
    return h * (1.0 + jnp.cos(ι)**2), h * (-2.0 * jnp.cos(ι))

def _compute_a_jax_compiled_simple(δ, α, ψ, h, ι, Ω, φ0, q, q_products, t, d_psr):
    """
    Simpler JAX version that matches the original numpy implementation exactly
    """
    # Convert to numpy arrays first and call the original implementation with JAX functions
    
    # Principal axes (using original code pattern)
    m, n = _principal_axes_jax(jnp.pi/2.0 - δ, α, ψ)
    gw_direction = jnp.cross(m, n)
    e_plus, e_cross = _polarisation_tensors_jax(m.T, n.T)
    hp, hx = _h_amplitudes_jax(h, ι)
    
    dot_product = 1.0 + q @ gw_direction.T  # Shape (N, M)
    
    # Amplitudes - exactly like numpy version
    M = δ.shape[0]
    Hij_plus = (hp * e_plus).reshape(9, M).T
    Hij_cross = (hx * e_cross).reshape(9, M).T
    
    Fplus = jnp.dot(Hij_plus, q_products)  # (M, N)
    Fcross = jnp.dot(Hij_cross, q_products)  # (M, N)
    
    # Phases - exactly like numpy version  
    earth_term_phase = jnp.outer(Ω, t).T + φ0  # Shape (T, M)
    phase_correction = Ω * dot_product * d_psr  # This works with proper broadcasting
    pulsar_term_phase = earth_term_phase.T.reshape(M, -1, 1) + phase_correction.T.reshape(M, 1, -1)
    
    # Trig terms
    cosine_terms = jnp.cos(earth_term_phase).reshape(-1, M, 1) - jnp.cos(pulsar_term_phase).transpose(1, 0, 2)
    sine_terms = jnp.sin(earth_term_phase).reshape(-1, M, 1) - jnp.sin(pulsar_term_phase).transpose(1, 0, 2)
    
    # Redshift per pulsar per source over time
    zplus = Fplus * cosine_terms
    zcross = Fcross * sine_terms
    z = (zplus + zcross) / (2 * dot_product.T)
    
    # Sum over sources
    a = jnp.sum(z, axis=1)
    
    return a

# JIT compile the function
_compute_a_jax_compiled = jit(_compute_a_jax_compiled_simple)

def _compute_a_batch_jax_compiled_simple(batch_δ, batch_α, batch_ψ, batch_h, batch_ι, batch_Ω, batch_φ0,
                                        q, q_products, t, d_psr):
    """
    Batch-vectorized JAX version that processes multiple universe realizations simultaneously
    
    Args:
        batch_*: Arrays with shape (num_realizations, M)
        q, q_products, t, d_psr: PTA parameters (same for all realizations)
    
    Returns:
        Array with shape (num_realizations, T, N)
    """
    
    # Use vmap to vectorize over the realization dimension (axis 0)
    vectorized_compute_a = vmap(_compute_a_jax_compiled_simple, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None))
    
    return vectorized_compute_a(batch_δ, batch_α, batch_ψ, batch_h, batch_ι, batch_Ω, batch_φ0,
                               q, q_products, t, d_psr)

# JIT compile the batch function  
_compute_a_batch_jax_compiled = jit(_compute_a_batch_jax_compiled_simple)

