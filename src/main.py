
import numpy as np
import sys 
from tqdm import tqdm

from PTA import PTA 
from BH_population import Universe
from calculate_gw import GW





def _get_a_for_universe_i(α,Ω_min,Ω_max,M,pulsars):

    universe_i = Universe(α,Ω_min,Ω_max,M) 
    


    #Given this universe and these pulsars, what is a(t)?
    SGWB = GW(universe_i,pulsars)
    a = SGWB.compute_a()

    a1 = a[:,0]
    a2 = a[:,1]


    #1D product
    ac = a1[0]
    product = ac*a2 


    #2D product
    outer_product = np.outer(a1,a2)

    return product,outer_product




def pipeline(Tobs,dt,pulsar_seed,num_realisations,α,Ω_min,Ω_max,M):

    #Choose the pulsars
    pulsars = PTA(Tobs=Tobs,dt=dt,seed=pulsar_seed)
    num_times = len(pulsars.t)

    array_1D = np.zeros((num_times,num_realisations)) #one timeseries for every seed. We will then average over this
    array_2D = np.zeros((num_times,num_times)) #2D grid for a running sum

    for i in tqdm(range(num_realisations)):
        product,outer_product = _get_a_for_universe_i(α,Ω_min,Ω_max,M,Tobs,dt,pulsar_seed)

    array_1D[:,i] = product
    array_2D += outer_product

    






#some useful quantities
year = 3.154e7 # in seconds
week = 604800  # in seconds

#Read in command line arguments
Tobs_years       = float(sys.argv[1])
dt_weeks         = float(sys.argv[2])
pulsar_seed      = int(sys.argv[3])
num_realisations = int(sys.argv[4])
M                = int(float(sys.argv[5]))
α                = float(sys.argv[6])



#Any necessary conversions
Tobs        = Tobs_years*year 
dt          = dt_weeks * week
Ω_min       = 1/(Tobs) 
Ω_max       = 1/(dt)

#Run the pipeline 
print(f"Creating {num_realisations} realisations of the stochastic GW background")

print(f"Parameters of the BH probability distribution are α ={α}, Ω_min= {Ω_min}, Ω_max= {Ω_max}")

pipeline(Tobs,dt,pulsar_seed,num_realisations,α,Ω_min,Ω_max,M)