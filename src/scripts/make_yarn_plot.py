import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import paths
from phases import *
from samplers import *


# Define colorblind-friendly colors (CBF)
# Source: https://arxiv.org/abs/2107.02270
CBF_COLORS = [
	"#3F90DA",  # blue
	"#FFA90E",  # orange
	"#BD1F01",  # red
	"#94A4A2",  # gray
	"#832DB6",  # purple
	"#A96B59",  # brown
	"#E76300",  # orange
	"#B9AC70",  # tan
	"#717581",  # gray
	"#92DADD",  # light blue
]

# Use colorblind-friendly colors as default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CBF_COLORS)


np.random.seed(42) # set a seed so the plots and data output are reporducible for showyourwork!

# this will make a plot with the number of orbits per panel and side of plots along one side
plotit = 1  # set to 0 to skip plot generation
norbs_plot = 10
nstars_plot = 4  # you will make a plot with nstars_plot by nstars_plot panels

tyb = dict(color='black', fontsize=6)


def draw_iwa(ax, iwa):
	from matplotlib.patches import Circle
	for r in iwa:
		# ax.add_patch(Circle((0, 0), radius=r, alpha=0.2))
		ax.add_patch(
			Circle(
				(0, 0), radius=r, facecolor='none', edgecolor='k', ls='--',
				lw=0.5, alpha=1.0
			)
		)


# number of planet sims per star
nplanets = 10

# number of epochs to sample along the orbit of the planet
norbitsample = 1000

# IWA ranges
# inner working angles (in mas) to compute betamax at
iwa = np.linspace(20, 120, 5)

# read in the stars
from astropy.io import ascii

fname = "2646_NASA_ExEP_Target_List_HWO_Table.csv"
t = ascii.read(paths.data / fname, header_start=1, data_start=2)

nstars = len(t)
print(f'# read in {nstars} stars from {fname}')

# set up output arrays
betamin = np.zeros((nplanets, nstars, iwa.size))
betamax = np.zeros((nplanets, nstars, iwa.size))

if plotit:

	pad_inches = 0.025
	fig, axes = plt.subplots(
		nstars_plot,
		nstars_plot,
		figsize=(7 - 2 * pad_inches, 7 - 2 * pad_inches)
	)

	ax = axes.flatten()
	# fig.suptitle(f'Inner Working Angles = {iwa} mas')

	# remove the x and y ticks
	for a in ax:
		a.set_xticks([])
		a.set_yticks([])
		a.set_aspect('equal')

	# fig.subplots_adjust(wspace=0, hspace=0)

cax = 0 # counter for the grid of subplots - this increases by 1 for each star AND also counts the star number for output array

for l in t: # loop over all the stars in the table
	star = l['hip_id']    		# name
	d = l['sy_dist']      		# parsecs
	Vmag = l['sy_vmag']   		# V magnitude
	mass = l['st_mass']   		# stellar mass
	hz = l['st_eei_angsep'] 	# Habitable zone distance (mas)

	# Set options for text boxes
	textbox_props = dict(
		boxstyle='square,pad=0.25',
		# linewidth=0.5,
		facecolor='white',
		edgecolor='none',
		alpha=0.8,
	)

	if plotit and (cax < (nstars_plot * nstars_plot)):
		ax[cax].set_xlim(-2*hz, 2*hz)
		ax[cax].set_ylim(-2*hz, 2*hz)
		draw_iwa(ax[cax], iwa)
		ax[cax].text(0.05, 0.95, star, ha='left', va='top', transform=ax[cax].transAxes, fontsize=6, weight='bold', bbox=textbox_props)
		ax[cax].text(0.05, 0.12, rf'$d_*$ = {d:.1f} pc', ha='left', va='bottom', transform=ax[cax].transAxes, bbox=textbox_props, **tyb)
		ax[cax].text(0.05, 0.05, rf'HZ = {hz:.1f} mas', ha='left', va='bottom', transform=ax[cax].transAxes, bbox=textbox_props, **tyb)

	# draw e and i from these distributions
	esamp = sample_e(nplanets)
	isamp = sample_i(nplanets) * u.deg
	anodesamp = np.random.random_sample((nplanets,)) * 360*u.deg
	wsamp = np.random.random_sample((nplanets,)) * 360*u.deg

	for n in np.arange(nplanets):

		# set up our orbit
		# CAUTION! you need to use Kepler's laws and know M1 and M2 to set P and a consistently :)
		#
		P = 1.0 * u.year
		tperi = 2050. * u.year
		a = hz  # in mas
		e = esamp[n]
		i = isamp[n]
		w = wsamp[n]
		anode = anodesamp[n]

		if plotit and (n < norbs_plot) and (cax < (nstars_plot*nstars_plot)):
			epochs = np.linspace(tperi, tperi+P, 100, endpoint=True)
			# draw the orbit
			_, _, Xorb, Yorb, Zorb, _, _, _ = kep3d(epochs, P, tperi, a, e, i, w, anode)
			ax[cax].plot(Xorb, Yorb)


	cax += 1 # increase figure number

if plotit:
	plt.draw()
	fig.tight_layout(pad=0)
	file_path = paths.figures / 'ball_of_yarn.pdf'
	plt.savefig(
		file_path,
		bbox_inches="tight",
		pad_inches=pad_inches,
	)
	plt.show()
