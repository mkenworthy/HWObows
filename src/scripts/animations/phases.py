import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@u.quantity_input
def kepler_solve(e, M:u.rad, derror=1e-5):
    """Kepler's equation solver using Newtons's method. Slow but robust.

    Args:
        e   eccentricity
        M   Mean anomaly (2pi t / P) since periapse

    Return:
        E   eccentric anomaly
        v   True anomaly

    Written:
        Matthew Kenworthy, 2010

    Text below taken from Terry R. McConnell's code
    at http://barnyard.syr.edu/quickies/kepler.c

    Kepler's equation is the transcendental equation

    E = M + e sin(E)

    where M is a given angle (in radians) and e is a number in the
    range [0,1). Solution of Kepler's equation is a key step in
    determining the position of a planet or satellite in its
    orbit.

    Here e is the eccentricity of the elliptical orbit and M is
    the "Mean anomaly." M is a fictitious angle that increases at a
    uniform rate from 0 at perihelion to 2Pi in one orbital period.

    E is an angle at the center of the ellipse. It is somewhat
    difficult to describe in words its precise geometric meaning --
    see any standard text on dynamical or positional astronomy, e.g.,
    W.E. Smart, Spherical Astronomy. Suffice it to say that E
    determines the angle v at the focus between the radius vector to
    the planet and the radius vector to the perihelion, the "true
    anomaly" (the ultimate angle of interest.) Thus, Kepler's equation
    provides the link between the "date", as measured by M, and the
    angular position of the planet on its orbit.

    The following routine implements the binary search algorithm due
    to Roger Sinnott, Sky and Telescope, Vol 70, page 159 (August 1985.)
    It is not the fastest algorithm, but it is completely reliable.

    you generate the Mean anomaly M convert to the Eccentric anomaly E
    through Kepler and then convert to the true anomaly v

    cos v = (cos E - e) / (1 - e.cos E)
    """

    #   solving  E - e sin(E) - M = 0
    scale = np.pi / 4.

    # first guess of E as M
    E = M

    # calculate the residual error
    R = E - (e * np.sin(E))*u.rad

    while True:

        if np.allclose(R.value, M.value, rtol=derror):
            break

        # where M is greater than R, add scale, otherwise subtract
        # scale

        sign = (M > R)

        # sign is 0 or 1
        E = E - (scale * (1 - (2*sign)))*u.rad

        R = E - (e * np.sin(E))*u.rad
        scale = scale / 2.0

    # calaculate the true anomaly from the eccentric anomaly
    # http://en.wikipedia.org/wiki/True_anomaly
    v = 2. * np.arctan2(np.sqrt(1+e) * np.sin(E/2) , np.sqrt(1-e) * np.cos(E/2))

    return E, v


@u.quantity_input
def euler(anode:u.rad, omega:u.rad, i:u.rad):
    """
    Build a 3D Euler rotation matrix of orbital elements

    Args:
        omega (float): the longitude of the periastron (u.angle)
        anode (float): ascending node angle (u.angle)
        i (float): inclination of the orbit (u.angle)

    Returns:
        Mat:   (3x3) rotation matrix

    Written:
        Matthew Kenworthy, 2017

    Taken from "Lecture Notes on Basic Celestial Mechanics" by Sergei
    A. Klioner (2011) celmech.pdf page 15
    """

    can  = np.cos(anode)
    san  = np.sin(anode)
    com  = np.cos(omega)
    som  = np.sin(omega)
    ci   = np.cos(i)
    si   = np.sin(i)

    e1 =  can*com - san*ci*som
    e2 = -can*som - san*ci*com
    e3 =  san*si
    e4 =  san*com + can*ci*som
    e5 = -san*som + can*ci*com
    e6 = -can*si
    e7 =  si*som
    e8 =  si*com
    e9 =  ci

    Mat = np.array([[e1, e2, e3],
                    [e4, e5, e6],
                    [e7, e8, e9]])

    return(Mat)

@u.quantity_input
def kep3d(epoch:u.year, P:u.year, tperi:u.year, a, e, inc:u.deg, omega:u.deg, anode:u.deg, derror=1e-6):
    """
    Calculate the position and velocity of an orbiting body

    Given the Kepler elements for the secondary about the primary
    and in the coordinate frame where the primary is at the origin

    Args:
        epoch (np.array):  epochs to evaluate (u.time)
        P (np.array): orbital period (u.time)
        tperi (float): epoch of periastron (u.time)
        a (float): semi-major axis of the orbit
        e (float): eccentricity of the orbit
        inc (float): inclination of the orbit  (u.angle)
        omega (float): longitude of periastron (u.angle)
        anode (float): PA of the ascending node (u.angle)

    Returns:
       X,Y, Xs,Ys,Zs, Xsv,Ysv,Zsv

    Output frame has X,Y in computer plotting coordinates
    i.e. X is to the right, increasing (due West)

    Primary body is fixed at the origin.

    X,Y (float): 2D coordinates of in plane orbit with periapse
                 towards the +ve X axis.

    Xs,Ys,Zs (float): The 3D coordinates of the secondary body
        in the Position/velocity coords frame.

    Xsv, Ysv, Zsv (float): The 3D velocity of the secondary body
        in the Position/velocity coords frame.

    The 3D axes are NOT the usual right-handed coordinate frame. The
    Observer is located far away on the NEGATIVE Z axis. This is done
    so that +ve Zsv gives positive velocities consistent with most
    astronomers idea of redshift being positive velocity values.


    Sky coords         Computer coords   Position/velocity coords

      North                   Y                +Y    +Z
        ^                     ^                 ^   ^
        |                     |                 |  /
        |                     |                 | /
        |                     |                 |/
        +-------> West        +-------> X       +-------> +X
                                               /
                                              /
                                             /
                                           -Z

    +Y is North, +X is West and +Z is away from the Earth
    so that velocity away from the Earth is positive

    NOTE: Right Ascension is INCREASING to the left, but the
    (X,Y,Z) outputs have RA increasing to the right, as seen
    in the Computer coords. This is done to make plotting easier
    and to remind the user that for the sky coords they need
    to plot (-X,Y) and then relabel the RA/X axis to reflect
    what is seen in the sky.

    Taken from ``Lecture Notes on Basic Celestial Mechanics''
    by Sergei A. Klioner (2011) page 22
    http://astro.geo.tu-dresden.de/~klioner/celmech.pdf

    Note that mean motion is defined on p.17 and that
    (P^2/a^3) = 4pi^2/kappa^2 and that the document is missing
    the ^2 on the pi.

    Coordinate frames are shown below.
   Written:
        Matthew Kenworthy, 2017

    """

    # mean motion n
    n = 2 * np.pi / P

    # at epoch = tperi, mean anomoly M is 0
    #
    # Y = time since epoch periastron
    Y = epoch - tperi

    # mean anomaly M varies smoothly from 0 to 2pi every orbital period
    # convert Y to angle in radians between 0 and 2PI
    Mt = Y / P
    M  = (2 * np.pi * (Mt - np.floor(Mt)))*u.radian

    # calc eccentric anomaly E
    (E,v) = kepler_solve(e, M, derror)

    # calculate position and velocity in the orbital plane
    cE = np.cos(E)
    sE = np.sin(E)
    surde = np.sqrt(1 - (e*e))

    X  = a * (cE - e)
    Y  = a * surde * sE

    Xv = -(a * n * sE) / (1 - e * cE)
    Yv =  (a * n * surde * cE) / (1 - e * cE)

    # calculate Euler rotation matrix to get from orbital plane
    # to the sky plane

    mat = euler(anode, omega, inc)

    # rotate the coordinates from the orbital plane
    # to the sky projected coordinates

    # TODO we lose the dimensionality of X and Y and it
    # needs to be put back artificially
    # problem with building the np.array below and putting the Quantity through
    # the np.dot() routine

    (Xe, Ye, Ze)    = np.dot(mat, np.array([X.value,Y.value,np.zeros_like((X.value))]))
    (Xev, Yev, Zev) = np.dot(mat, np.array([Xv.value,Yv.value,np.zeros_like((X.value))]))

    Xs = -Ye * X.unit
    Ys =  Xe * X.unit
    Zs =  Ze * X.unit
    Xsv = -Yev * Xv.unit
    Ysv =  Xev * Xv.unit
    Zsv =  Zev * Xv.unit

    # To transform from the Kep3D code coordinate system to the
    # celestial sphere, where:
    # Y is North, X is West and Z is away from the Earth
    # we have
    #   X,Y,Z,Xdot,Ydot,Zdot = -Ys, Xs, Zs, -Yv, Xv, Zv

    return(X,Y,Xs,Ys,Zs,Xsv,Ysv,Zsv)


def xyztoscatter(x,y,z):
  ''' convert the x,y,z, position of the planet to the scattering angle
  assume direction towards earth is along negative z

  \
   \
    \ 
     \
  x   \
       \
        \
  ----z---
  '''
  # calculate hypotenuse of (0,x,y) triangle
  xy = np.sqrt(x*x+y*y)

  # acute angle in xy and z right angled triangle
  return np.arctan2(xy,z)


def rot(x,y,th):
  xr =  x*np.cos(th) + y*np.sin(th)
  yr = -x*np.sin(th) + y*np.cos(th)
  return xr,yr



def moon(ax,xc,yc,scatterang=np.pi/2,ang=30,rad=2):
    '''draw a partially filled moon at x,y with percent filled and ang rotation from horizontal
    ang - angle in radians rotated clockwise from x axis'''
    from matplotlib.patches import Polygon
    npoi = 24 # points around circle
    th = np.linspace(0,np.pi,npoi)
    x = rad*np.cos(th)
    y = rad*np.sin(th)

    # make the whole disk
    xdisk = np.append(x,x[::-1])
    ydisk = np.append(y,-y[::-1])
    # rotate and translate to the final position...

    # the crescent phase
    xill = x
    yill = y*np.cos(scatterang)
    xcresc = np.append(x,x[::-1])
    ycresc = np.append(y,-yill[::-1])

    # rotate both (yes, even the circular disk because points will coadd)
    (xdisk,ydisk) = rot(xdisk,ydisk,ang)
    (xcresc,ycresc) = rot(xcresc,ycresc,ang)

    xdisk += xc
    ydisk += yc

    xcresc += xc
    ycresc += yc

    # generate the disk and illuminated area of the crescent
    disk = Polygon(np.transpose(np.array([xdisk,ydisk])),closed=True,edgecolor='black',facecolor='black')
    ax.add_patch(disk)

    cresc = Polygon(np.transpose(np.array([xcresc,ycresc])),closed=True,edgecolor='none',facecolor='white')
    ax.add_patch(cresc)


if __name__ == "__main__":
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    (ax1, a2, a3, a4) = axes.flatten()
    ax1.set_xlim(-10,10)
    ax1.set_ylim(-10,10)
    moon(ax1,3,1)
    ax1.set_title('testing the moon() routine')
    plt.show()
