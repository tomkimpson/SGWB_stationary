
import matplotlib.pyplot as plt
import numpy as np



def plot_1d(t,array_1D,show_fig=True,save_fig=False):



    #Setup the figure
    h,w = 8,16
    rows = 3
    cols = 1
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharey=False,sharex=True)
    
    #t = t-t[0]
    year = 365*24*3600
    tplot = t / year


    for i in range(array_1D.shape[-1]):
        ax1.plot(tplot,array_1D[:,i])



    av = np.mean(array_1D,axis=1)
    sd = np.sqrt(np.var(array_1D,axis=1))


    ax2.plot(tplot,av)
    ax3.plot(tplot,sd)

    
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')



    ax2.set_xlim(tplot[1],tplot[-1])



    plt.subplots_adjust(hspace=0.05)



    # #Tidy it up
    fs=20
    ax3.set_xlabel(r'$\tau$ [years]', fontsize=fs)
   


    ax1.set_ylabel(r'$L^{(j)}(\tau)$', fontsize=fs) 
    ax2.set_ylabel(r'$L(\tau)$', fontsize=fs) 
    ax3.set_ylabel(r'$ \sigma \left[ L(\tau) \right]$', fontsize=fs) 

    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelsize=fs-4)
        ax.yaxis.set_tick_params(labelsize=fs-4)
        ax.set_xscale('log')

    if save_fig:
        import os
        from datetime import datetime
        os.makedirs('outputs', exist_ok=True)
        # Generate unique ID using timestamp
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'outputs/plot_1d_{unique_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()



from labellines import labelLines 
def plot_2d(t,array_2D,plot_points=False,num_contours=100,show_fig=True,save_fig=False):





    #Setup the figure
    h,w = 8,8
    rows = 1
    cols = 1
    fig, ax1 = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w))



    xx_years = t / (365*24*3600) #cadence = 1 day
    yy_years = t / (365*24*3600)


    ax1.contour(xx_years, yy_years, array_2D, cmap='viridis',levels=num_contours)




    def fixed_contour(x,c):
        return x + c 

    cs = [0,1,2,3]
    ls = ['solid','dotted','dashed','dashdot']
    for i in range(len(cs)):

        y1 = fixed_contour(xx_years,cs[i])
        ax1.plot(xx_years,y1,c='0.5', linestyle=ls[i],label=str(cs[i])+'yr')



        y1 = fixed_contour(xx_years,-cs[i])
        ax1.plot(xx_years,y1,c='0.5', linestyle=ls[i],label=str(cs[i])+'yr')

    fs=20
    ax1.xaxis.set_tick_params(labelsize=fs-4)
    ax1.yaxis.set_tick_params(labelsize=fs-4)

    ax1.set_ylim(0,10)
    ax1.set_xlim(0,10)

    ax1.set_xlabel(r'$t$ [years]', fontsize=fs)
    ax1.set_ylabel(r"$t'$ [years]", fontsize=fs) 


    if plot_points:
        X,Y = np.meshgrid(xx_years,yy_years)
        ax1.scatter(X.flatten(),Y.flatten(),s=2,marker='x')

    
    lines=plt.gca().get_lines()
    xv = np.ones(len(lines))*5
    labelLines(lines,align=True,xvals=xv,fontsize=fs-6)

    if save_fig:
        import os
        from datetime import datetime
        os.makedirs('outputs', exist_ok=True)
        # Generate unique ID using timestamp
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'outputs/plot_2d_{unique_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()







