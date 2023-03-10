import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate as si
import paths
from hwo import *

CBF_COLORS = [
    "#5790fc",  # blue
    "#f89c20",  # orange
    "#e42536",  # red
    "#964a8b",  # purple
    "#9c9ca1",  # gray
    "#7a21dd",  # purple
]
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CBF_COLORS)

def make_features_plot_multi(betas,hab_zones,features,iwa_list = [1,2,3,4],eta_earth=0.24,d=6,
                       nbins=20,wavelength=700e-9,figure_filename="nplanets_vs_iwa.png",
                       save_fig=False,cmap_name='Spectral'):
    '''
    A function to make a multi-panneled plot of four different scattering features
    Showing the numner of accessible systems for the 
    low end, middle and upper end of the phase angle ranges of each feature, as a function of IWA. 

    betas - the maximum scattering angle parameter, here defined as beta = phase_angle_max - 90
    iwa_list - list of IWAs you want to use, shape [n_iwas]
    features - A dictionary of scattering features that includes the lower, middle and upper phase angle ranges (in degrees)
    hab_zones - list of habitable zone SMAs in mas, shape [n_systems]
    eta_earth - the assumption on the number of systems have interesting planets (basically a scale factor for the y-axis), scalar
    wavelength - wavelength you want to consider the IWA for in m
    d - diameter of your detelescope in m
    cmap_name - The colormap you want to use to color the lines
    n_bins - the number of bins to use when making hisotgram bins for the cdfs, scalar
    savefig - whether to save the figure or not, boolean
    filename - the filename of the figure you want to save, string
    '''

    # if ax is None:
    #     fig, axis = plt.subplots(1,1)
    # else: 
    #     axis=ax

    fig,axes = plt.subplots(2,2,figsize=(12,5),sharex=True,sharey=True)
    axes = axes.flatten()

    for k,feature in enumerate(features):
        lows = []
        meds = []
        highs = []

        for i in range(len(iwa_list)):
        
            #This helps identify which systems aren't detected at all. 
            # There are easier ways, but this is legacy and works
            beta_max = np.degrees(np.arccos(iwa_list[i]/(hab_zones)))

            # getting data of the histogram
            count, bins_count = np.histogram(betas[:,:,i].flatten(), bins=np.linspace(90,180,nbins))
            # finding the PDF of the histogram using count values
            pdf = count / sum(count)
            # using numpy np.cumsum to calculate the CDF
            # We can also find using the PDF values by looping and adding
            cdf = np.cumsum(pdf)

            normalized_cdf = (1-cdf)*np.sum(np.isfinite(beta_max))
            
            #Deal with angles below 90
            these_angles = np.array(features[feature])
            # print(these_angles)
            for m,angle in enumerate(these_angles): 
                if angle < 90:
                    these_angles[m] = 90+(90-(angle))

            #Interpolate the inverse cdfs to the specified angles
            interp_cdf = si.interp1d(bins_count[1:],normalized_cdf,fill_value = 'extrapolate',kind='slinear')(these_angles)
            
            #Append the values to lists for future plotting
            lows.append(interp_cdf[0])
            meds.append(interp_cdf[1])
            highs.append(interp_cdf[2])

        ### Fixed Colors
        # axes[k].plot(iwa_list,meds,color='C{:d}'.format(k),label=feature)
        # axes[k].plot(iwa_list,meds,color='C{:d}'.format(k),label="Approximate Peak of Feature")
        # axes[k].plot(iwa_list,lows,'--',color='C{:d}'.format(k),label="Start of Feature")
        # axes[k].plot(iwa_list,highs,'-.',color='C{:d}'.format(k),label="Beyond Peak of Feature")
        # if k == 1:
        #     axes[k].text(80,100,feature,color='C{:d}'.format(k))
        # elif k == 3:
        #     axes[k].text(70,20,feature,color='C{:d}'.format(k))
        # else: 
        #     axes[k].text(70,80,feature,color='C{:d}'.format(k))

        ### Colors according to a colormaps
        cmap = mpl.cm.get_cmap(cmap_name)
        
        low_rgba = cmap((these_angles[0]-90)/90)
        med_rgba = cmap((these_angles[1]-90)/90)
        high_rgba = cmap((these_angles[2]-90)/90)
        
        axes[k].plot(iwa_list,lows,'--',color=low_rgba,label="Start of Feature")
        axes[k].plot(iwa_list,meds,color=med_rgba,label="Approximate Peak of Feature")
        axes[k].plot(iwa_list,highs,'-.',color=high_rgba,label="Past Feature Peak")

        #Location of the text labels. Hardcoded :( 
        if k == 1:
            axes[k].text(80,100,feature,color=med_rgba)
        elif k == 3:
            axes[k].text(70,20,feature,color=med_rgba)
        else: 
            axes[k].text(70,80,feature,color=med_rgba)
            
        #Where do we set the axis labels? - Deprecated, see below. 
        # if (k == 2) or (k == 0):
            # axes[k].set_ylabel(r"Number of systems")
        # if (k == 2) or (k == 3):
            # axes[k].set_xlabel(r"IWA (mas)")

        #Set the ylimits
        axes[k].set_ylim(0,170)

        #Only make a in the first pot
        if k==0 :
            axes[k].legend()

        #Manually setting the x-tick locations
        xticks = np.arange(20, np.around(iwa_list[-1],decimals=1)+10, 10)
        axes[k].set_xticks(xticks)

        #Manually setting the y-tick ocations
        yticks = np.arange(0, 180, 20)
        axes[k].set_yticks(yticks)

        ## Setup the twin axes
        if (k == 1) or (k == 3):
            twiny = axes[k].twinx()
            twiny.set_ylim(axes[k].set_ylim())
            # twiny.set_ylabel(r"Number of planets $\eta_\oplus$={}".format(eta_earth))
            twiny.set_yticks(yticks)
            twiny.set_yticklabels(["{:.0f}".format(x*eta_earth) for x in yticks])

        if (k == 0) or (k== 1):
            twinx = axes[k].twiny()
            twinx.set_xlim(axes[k].set_xlim())
            new_xticks = [x*(wavelength/d*206265*1000) for x in np.arange(1,7)]
            twinx.set_xticks(new_xticks)
            twinx.set_xticklabels(["{:.0f}".format((x/(wavelength/d*206265*1000))) for x in new_xticks])
            # twinx.set_xlabel(r"IWA ($\lambda/D$, $\lambda={:.0f}nm$)".format(wavelength*1e9))

    #Squeeze the plots together
    fig.subplots_adjust(wspace=0, hspace=0)

    #### Setup the x and y axis labels. 
    ylabel_axis = fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    plt.ylabel(r"Number of systems",labelpad=25)
    plt.xlabel(r"IWA (mas)",labelpad=25)
    ylabel_axis.set_xticks([])
    ylabel_axis.set_yticks([])

    twinx = ylabel_axis.twiny()
    plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    twinx.set_xlabel(r"IWA ($\lambda/D$, $\lambda={:.0f}nm$)".format(wavelength*1e9),labelpad=25)
    twinx.set_xticks([])
    twinx.set_yticks([])

    twiny = ylabel_axis.twinx()
    plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    twiny.set_ylabel(r"Number of planets $\eta_\oplus$={}".format(eta_earth),labelpad=25)
    twiny.set_xticks([])
    twiny.set_yticks([])

    if save_fig:
        fig.savefig(paths.figures / figure_filename,dpi=200)

if __name__ == "__main__":
    #############
    ### Setup ###
    #############

    ##Read in the ExEP Target List
    df = pd.read_csv(paths.data / '2646_NASA_ExEP_Target_List_HWO_Table.csv')
    #The angular separation of the Earth Equivalent Instellation
    hab_zones = np.array(df['EEIDmas'].values[1:],dtype=float) #in mas

    ##Some assumed parameters (now imported from hwo.py)
 #   d = 6 #telescope diameter
 #   wavelength = 600e-9

    ## Read in the maximum and minimum scattering angles as 
    ## calculated by the dynamical simulations
    ## The files with the 3 suffix include the wider range of 
    ## iwa that we want for this step
    iwa = np.load('iwa_all3.npz')
    phase_max = iwa['betamax']
    phase_min = iwa['betamin']
    iwa_list =  iwa['iwa']
#    phase_max=np.load(paths.data / "betamax3.npy")
#    phase_min=np.load(paths.data / "betamin3.npy")
#    iwa_list=np.load(paths.data / "iwa3.npy")
    # Note: We pass in beta to the plotting function here, based on phase_max
    # Beta is an intermediate parameter used here. 
    # The maximum scattering angle is equal to 90+beta degrees
    # The minimum scattering angle is equal to 90-beta degrees

    ## Here we define the names of the features that will go in each plot, 
    ## As well as the lower range, middle of the range and upper range of the features'
    ## phase angles (in degrees)
    features = {r'Rainbow': (127,138,158),
            r"Rayleigh": (90,110,130),
            r"Ocean Glint": (50,30,10),
            r"Glory": (10,5,0),
            }

    make_features_plot_multi(phase_max,hab_zones, features,iwa_list=iwa_list,
                   wavelength=wavelength,save_fig=True,cmap_name='viridis_r')
