"""
Create Figure 5: Grid of orbits with different inclinations and IWAs.
"""

from astropy.time import Time
from matplotlib.patches import Arrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from utils import paths
from utils.orbfunc import xyz_position, phase


##########################################################################################



params = {
    'time periapsis passage':Time('2021-09-29T00:00:00', format='fits'),
    'orbital period':1*u.day,
    'semi-major axis':1*u.au,
    'eccentricity':0.0,
    'argument of periapsis':np.pi*0/180*u.rad,
    'periapsis precession':0*u.rad/u.s,
    'inclination':np.pi*75/180*u.rad,
    'longditude of ascending node':np.pi*90/180*u.rad
}

time = params['time periapsis passage'] - 0.5*u.day + np.arange(0,1,0.001)*u.day

##########################################################################################

def setup_plot(fig, i, oplane=False, losplane=False):

    ax = fig.add_subplot(4, 3, i, projection='3d')
    ax.set_proj_type('ortho')
    ax.margins(-0.499)

    # plot orbit
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'k-', lw=2)

    # plane of los
    if losplane==True:
        lossurfx, lossurfy = np.meshgrid(np.linspace(-0.65,0.65,10),np.linspace(-0.65,0.65,10))
        x1 = ax.plot_surface(lossurfx, lossurfy, np.zeros((10,10)), color='g', edgecolor=(0,0,0,0.15), alpha=0.1, label='Plane of Nodes')

        ### Trying to catch a version dependent error in matplotlib, may not work for all versions ###
        try:
            x1._facecolors2d = x1._facecolor3d
            x1._edgecolors2d = x1._edgecolor3d
        except:
            try:
                x1._facecolors2d = x1._facecolor3d
                x1._edgecolors2d = x1._edgecolor3d
            except:
            	pass
        ###############################################################################################


    # plane of orbit
    if oplane==True:
        A = np.ones((len(time),3)); A[:,0] = xyz[:,0]; A[:,1] = xyz[:,1]; A = np.matrix(A)
        B = np.matrix(xyz[:,2]).T

        fit = ((A.T * A).I * A.T * B)

        surfx, surfy = np.meshgrid(np.linspace(np.min(xyz[:,0]),np.max(xyz[:,0]),10),np.linspace(np.min(xyz[:,1]),np.max(xyz[:,1]),10))
        x2 = ax.plot_surface(surfx, surfy, fit[0,0]*surfx + fit[1,0]*surfy + fit[2,0], color='m', alpha=0.1, label='Plane of Orbit')

        ### Trying to catch a version dependent error in matplotlib, may not work for all versions ###
        try:
            x2._facecolors2d = x2._facecolor3d
            x2._edgecolors2d = x2._edgecolor3d
        except:
            try:
                x2._facecolors2d = x2._facecolor3d
                x2._edgecolors2d = x2._edgecolor3d
            except:
            	pass
        ###############################################################################################

    # mark points
    ax.plot3D([0],[0],[0], 'C1*', label='Star', markersize=15)

    ax.set_xlim(0.7,-0.7)
    ax.set_ylim(-0.7,0.75)
    ax.set_zlim(-0.2,0.2)
    ax.axis('off')

    return ax

##########################################################################################

xyz = xyz_position(time, params).to(u.au).value

fig = plt.figure(figsize=(15,20)) #plt.figaspect(1)*2)

index_an = np.argmin(abs(phase(time, params)-params['argument of periapsis'].to(u.rad).value/(2*np.pi)))
pend = np.arctan2(xyz[index_an,1], -xyz[index_an,0])
if pend<0: pend +=np.pi*2

### Plot 1: inc ###
ax = setup_plot(fig, 2, oplane=True, losplane=True)

# mark inc
di = 0.5
inc = np.pi/2 - params['inclination'].to(u.rad).value
ax.plot3D(di*np.cos(np.linspace(0,inc,100))*np.sin(pend)+xyz[index_an,0], di*np.cos(np.linspace(0,inc,100))*np.cos(pend)+xyz[index_an,1], di*np.sin(np.linspace(0,inc,100)), 'r-')
ax.plot3D([xyz[index_an,0], 1.5*di*np.cos(inc)+xyz[index_an,0]],[xyz[index_an,1], xyz[index_an,1]], [0,1.5*di*np.sin(inc)], 'r--')
ax.plot3D([xyz[index_an,0], xyz[index_an,0]+1.5*di*np.sin(pend)],[xyz[index_an,1], xyz[index_an,1]+1.5*di*np.cos(pend)], [0,0], 'r--')
ax.text(xyz[index_an,0]+0.4,xyz[index_an,1]-0.4, -di/8, '$90^\circ-i$', None, fontsize=20, va='center', ha='center')

# mark LOS arrow
ax.plot3D([-0.65,0], [0, 0], [0, 0], 'k<-', label='Line of Sight (to Earth)', markersize=1, zorder=-10)
ax.plot3D([-0.65], [0], [0], 'k', markersize=15, marker=(3,0,90))
ax.view_init(elev=5, azim=80)

# legend 1
ax = fig.add_subplot(4, 3, 1, projection='3d')
ax.axis('off')
ax.plot3D([],[],[], 'C1*', label='Star', markersize=15)
ax.plot3D([],[],[], 'k-<', label='Line of Sight (to Earth)')
ax.plot3D([],[],[], 'ko', label='Mask', alpha=0.3)
ax.plot([],[],[], ' ', label='Inclination')
ax.legend(framealpha=0, loc=2, fontsize=16)
ax.text2D(0.1,0.65, '$i$', transform=ax.transAxes, fontsize=18, fontweight='bold')

# legend 2
ax = fig.add_subplot(4, 3, 3, projection='3d')
ax.axis('off')
ax.scatter([],[], c='#5790fc', marker='o', label=r'$\phi = 90\degree$')
ax.scatter([],[], c='#964a8b', marker='o', label=r'$\phi_{max}$')
ax.scatter([],[], c='#7a21dd', marker='o', label=r'$\phi_{min}$')
ax.legend(framealpha=0, loc=2, fontsize=16)

### inc 1: mask 1 ###
ax = setup_plot(fig, 4)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=75, alpha=0.3)
ax.view_init(elev=0, azim=0)

### inc 1: mask 2 ###
ax = setup_plot(fig, 5)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=150, alpha=0.3)
ax.plot3D([xyz[500,0]], [xyz[500,1]], [xyz[500,2]], c='#5790fc', marker='o', markersize=10)
ax.plot3D([xyz[250,0]], [xyz[250,1]], [xyz[250,2]], c='#7a21dd', marker='o', markersize=10)
ax.plot3D([xyz[750,0]], [xyz[750,1]], [xyz[750,2]], c='#964a8b', marker='o', markersize=10)
ax.view_init(elev=0, azim=0)

### inc 1: mask 3 ###
ax = setup_plot(fig, 6)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=200, alpha=0.3)
ax.view_init(elev=0, azim=0)

### inc 2: mask 1 ###
ax = setup_plot(fig, 7)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=75, alpha=0.3)
ax.view_init(elev=-20, azim=0)

iaxes = inset_axes(ax,width=1,height=3,bbox_transform=ax.transAxes,bbox_to_anchor=(-0.2,0.5),loc=6)
iaxes.add_patch(Arrow(0.5,0.85,0,-0.7, width=0.6, color='k'))
iaxes.annotate('Inclination', (0.3, 0.55), color='k', fontsize=16, ha='center', va='center', rotation=90)
iaxes.axis('off')

### inc 2: mask 2 ###
ax = setup_plot(fig, 8)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=150, alpha=0.3)
ax.plot3D([xyz[500,0]], [xyz[500,1]], [xyz[500,2]], c='#5790fc', marker='o', markersize=10)
ax.plot3D([xyz[350,0]], [xyz[350,1]], [xyz[350,2]], c='#7a21dd', marker='o', markersize=10)
ax.plot3D([xyz[650,0]], [xyz[650,1]], [xyz[650,2]], c='#964a8b', marker='o', markersize=10)
ax.view_init(elev=-20, azim=0)

### inc 2: mask 3 ###
ax = setup_plot(fig, 9)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=200, alpha=0.3)
ax.view_init(elev=-20, azim=0)

### inc 3: mask 1 ###
ax = setup_plot(fig, 10)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=75, alpha=0.3)
ax.view_init(elev=-40, azim=0)

### inc 3: mask 2 ###
ax = setup_plot(fig, 11)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=150, alpha=0.3)
ax.plot3D([xyz[500,0]], [xyz[500,1]], [xyz[500,2]], c='#5790fc', marker='o', markersize=10)
ax.plot3D([xyz[360,0]], [xyz[360,1]], [xyz[360,2]], c='#7a21dd', marker='o', markersize=10)
ax.plot3D([xyz[640,0]], [xyz[640,1]], [xyz[640,2]], c='#964a8b', marker='o', markersize=10)
ax.view_init(elev=-40, azim=0)

iaxes = inset_axes(ax,width=3,height=1,bbox_transform=ax.transAxes,bbox_to_anchor=(0.5,0.0),loc=8)
iaxes.add_patch(Arrow(0.15,0.5,0.7,0, width=0.6, color='k'))
iaxes.annotate('Inner Working Angle', (0.5, 0.2), color='k', fontsize=16, ha='center', va='center')
iaxes.axis('off')

### inc 3: mask 3 ###
ax = setup_plot(fig, 12)
ax.plot3D([0],[0],[0], 'ko', label='Mask', markersize=200, alpha=0.3)
ax.view_init(elev=-40, azim=0)

plt.subplots_adjust(wspace=0.0, hspace=-0.4)
plt.savefig(paths.figures / 'figure-5-orb-grid.pdf')
