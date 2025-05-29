# SGWB_stationary
Work-in-progress exploration of the stationarity of the stochastic GW background.

Please run `src/main.py` to generate the relevant plots. Some example plots can be seen in `src/outputs/`

Please see `notebooks/demo.ipynb` for some additional discussion. 


### Additional physics notes

All GWs are taken to have the same (unit) amplitude that is decoupled from their frequency $\Omega$.

The PDF of $\Omega$ is taken to be a power law with lower limit $1/T_{\rm obs}$, upper limit $1/dt$ and exponent -3.0. These numbers can be played with.



### Additional computation notes

The code is written with JAX to enable JIT compilation and GPU processing.

For $M=10^4$ and $2 \times 10^4$ universe realisations, we can compute $\langle a^{(n)}(t) a^{(n')}(t')\rangle (\tau)$ is ~2min on a GPU. 
