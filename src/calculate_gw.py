


import numpy as np 

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
        earth_term_phase  = np.outer(self.Ω,self.t).T + + self.φ0 # Shape(T,M)
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

