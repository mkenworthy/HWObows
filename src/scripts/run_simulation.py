import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from phases import *
from samplers import *

from tqdm import tqdm
import time
import paths

np.random.seed(42) # set a seed so the plots and data output are reporducible for showyourwork!

# number of planet sims per star
nplanets = 1000

# number of epochs to sample along the orbit of the planet
norbitsample = 1000

# IWA ranges as selected for three possible iwa
op = snakemake.params['whichsim']
#op = sys.argv[1]
if op == 'iwa':
	iwa = np.array([21, 41, 62, 83])
	tag = ''
elif op == 'iwa2':
	iwa = np.linspace(20,120,5) # inner working angles (in mas) to compute betamax
	tag = '2'
elif op == 'iwa3':
	iwa = np.linspace(20,120,20)
	tag = '3'
else:
	print('not a valid command line argument. Stopping...')
	quit()
print(f'option selected was {op}')
# read in the stars
from astropy.io import ascii

fname = "2646_NASA_ExEP_Target_List_HWO_Table.csv"
t = ascii.read(paths.data / fname ,header_start = 1, data_start = 2)

nstars = len(t)
print(f'# read in {nstars} stars from {fname}')

# set up output arrays
betamin = np.zeros((nplanets,nstars,iwa.size))
betamax = np.zeros((nplanets,nstars,iwa.size))

cax = 0 # counter for the grid of subplots - this increases by 1 for each star AND also counts the star number for output array

for l in tqdm(t): # loop over all the stars in the table
	star = l['hip_id']    		# name
	d = l['sy_dist']      		# parsecs
	Vmag = l['sy_vmag']   		# V magnitude
	mass = l['st_mass']   		# stellar mass
	hz =  l['st_eei_angsep'] 	# Habitable zone distance (mas)

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

print(f'saving the data with command line option {op}')
np.savez_compressed(paths.data / f'iwa_all{tag}.npz', iwa=iwa, betamax=betamax, betamin=betamin)


	
