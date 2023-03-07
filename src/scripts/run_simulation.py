import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from phases import *
from samplers import *

from tqdm import tqdm
import time

np.random.seed(42) # set a seed so the plots and data output are reporducible for showyourwork!

# this will make a plot with the number of orbits per panel and side of plots along one side
plotit = 0 # set to 0 to skip plot generation
norbs_plot = 10
nstars_plot = 4 # you will make a plot with nstars_plot by nstars_plot panels

tyb = dict(color='black', fontsize=14)

def draw_iwa(ax, iwa):
	from matplotlib.patches import Circle
	for r in iwa:
		ax.add_patch(Circle((0,0),radius=r,alpha=0.2))

# number of planet sims per star
nplanets = 1000

# number of epochs to sample along the orbit of the planet
norbitsample = 1000

# IWA ranges
iwa = np.linspace(20,120,5) # inner working angles (in mas) to compute betamax at

# read in the stars
from astropy.io import ascii

fname = "2646_NASA_ExEP_Target_List_HWO_Table.csv"
t = ascii.read(fname ,header_start = 1, data_start = 2)

nstars = len(t)
print(f'# read in {nstars} stars from {fname}')

# set up output arrays
betamin = np.zeros((nplanets,nstars,iwa.size))
betamax = np.zeros((nplanets,nstars,iwa.size))

if (plotit):
	fig, axes = plt.subplots(nstars_plot, nstars_plot, figsize=(10,10))
	ax = axes.flatten()
	fig.suptitle(f'Inner Working Angles = {iwa} mas')
	# remove the x and y ticks
	for a in ax:
		a.set_xticks([])
		a.set_yticks([])
		a.set_aspect('equal')

	fig.subplots_adjust(wspace=0, hspace=0)

cax = 0 # counter for the grid of subplots - this increases by 1 for each star AND also counts the star number for output array

for l in tqdm(t): # loop over all the stars in the table
	star = l['hip_id']    		# name
	d = l['sy_dist']      		# parsecs
	Vmag = l['sy_vmag']   		# V magnitude
	mass = l['st_mass']   		# stellar mass
	hz =  l['st_eei_angsep'] 	# Habitable zone distance (mas)

#	print(f'# working on {star} {cax} of {nstars} at {d:.1f} parsecs with V={Vmag:.1f} and HZ radius={hz:.1f} mas')

	if (plotit) and (cax < (nstars_plot*nstars_plot)):
		ax[cax].set_xlim(-1.5*hz, 1.5*hz)
		ax[cax].set_ylim(-1.5*hz, 1.5*hz)
		draw_iwa(ax[cax],iwa)
		ax[cax].text(0.05, 0.95, star, ha='left', va='top', transform=ax[cax].transAxes, fontfamily='bold', fontsize=18)
		ax[cax].text(0.05, 0.05, f'HZ={hz:.1f} mas', ha='left', va='bottom', transform=ax[cax].transAxes, **tyb)


	# draw e and i from these distributions
	esamp = sample_e(nplanets)
	isamp = sample_i(nplanets) * u.deg
	anodesamp = np.random.random_sample((nplanets,)) * 360*u.deg 
	wsamp = np.random.random_sample((nplanets,)) * 360*u.deg 

#	for n in tqdm(np.arange(nplanets)):
	for n in np.arange(nplanets):
		#print(f'# sim number {n+1} of {nplanets}...')

		# set up our orbit
		# CAUTION! you need to use Kepler's laws and know M1 and M2 to set P and a consistently :)
		#
		P = 1.0 * u.year
		tperi = 2050. * u.year
		a = hz # in mas
		e = esamp[n]
		i = isamp[n]
		w = wsamp[n]
		anode = anodesamp[n]

		if (plotit) and (n < norbs_plot) and (cax < (nstars_plot*nstars_plot)):
			epochs = np.linspace(tperi, tperi+P, 100, endpoint='true')
			# draw the orbit
			_, _, Xorb, Yorb, Zorb, _, _, _ = kep3d(epochs,P,tperi,a,e,i,w,anode)
			ax[cax].plot(Xorb,Yorb)

		epochs = np.linspace(tperi, tperi+P, norbitsample, endpoint='true')

		# calc the orbit
		_, _, Xsa, Ysa, Zsa, _, _, _ = kep3d(epochs,P,tperi,a,e,i,w,anode)

		# calculate current projected separation
		rho = np.sqrt(Xsa*Xsa+Ysa*Ysa)

		# distance from star to planet
		rad_dist = np.sqrt(Xsa*Xsa+Ysa*Ysa+Zsa*Zsa)

		# calculate scattering angle
		scang = xyztoscatter(Xsa,Ysa,Zsa)

		# for each iwa, give True where in orbit you are behind the coroangraphic mask [rho.shape, iwa.shape]
		notvisible = (rho[:,np.newaxis] < iwa)

		# dummy array in preparation for max and min
		ttt = scang[:,np.newaxis] * np.ones(iwa.size)

		# for each iwa column, set the points behind the mask as NaN
		ttt[notvisible] = np.nan

		# get the largest and smallest scattering angles, NaN if the orbit is all behind the mask!
		scatmax = np.nanmax(ttt,axis=0).to(u.deg)
		scatmin = np.nanmin(ttt,axis=0).to(u.deg)
		betamax[n,cax] = scatmax
		betamin[n,cax] = scatmin

	cax+=1 # increase figure number

np.save('iwa2.npy',iwa)
np.save('betamax2.npy',betamax)
np.save('betamin2.npy',betamin)

if plotit:
	plt.draw()
	plt.savefig('ball_of_yarn.png',dpi=100)
	plt.savefig('ball_of_yarn.pdf')
	plt.show()

	
