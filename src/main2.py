






from BH_population import Universe
from PTA import PTA 
from calculate_gw import GW
import numpy as np



def get_a_for_universe_i(pulsar_seed):

    universe_i = Universe(alpha,Ω_min,Ω_max,M) #this is a realisation of the universe
    pulsars = PTA(Tobs=Tobs,dt=dt,seed=pulsar_seed)


    #Given this universe and these pulsars, what is a(t)?
    SGWB = GW(universe_i,pulsars)
    a = SGWB.compute_a_jax()

    a1 = a[:,0]
    a2 = a[:,1]


    #1D product
    ac = a1[0]
    product = ac*a2 


    #2D product
    outer_product = np.outer(a1,a2)

    return product,outer_product





#some useful quantities
year = 3.154e7 # in seconds
week = 604800  # in seconds



#Define the parameters for the power law over Ω
alpha = -3.0 #Exponent of the power law for the PDF of Ω
Ω_min = 1/(10*year) #lower bound on the Ω power law distribution. Set by Tobs
Ω_max = 1/(week)  #upper bound on the Ω power law distribution. Set by dt
M = int(1e4)


#Observation period
Tobs = 10*year 
dt = 1*week



pulsars = PTA(Tobs=Tobs,dt=dt,seed=1)
num_times = len(pulsars.t)



from tqdm import tqdm

num_realisations = 1000

array_1D = np.zeros((num_times,num_realisations)) #one timeseries for every seed. We will then average over this
array_2D = np.zeros((num_times,num_times)) #2D grid for a running sum

for i in tqdm(range(num_realisations)):
    product,outer_product = get_a_for_universe_i(pulsar_seed=1)

    array_1D[:,i] = product
    array_2D += outer_product





from plotting import plot_1d,plot_2d



plot_1d(pulsars.t,array_1D,show_fig=False,save_fig=True)
plot_2d(pulsars.t,array_2D,show_fig=False,save_fig=True)