from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import *

def Z_aleatoire_croissant(taille,hmin,hmax):
    #z1 = np.array([uniform(hmin,hmax) for i in range (taille)])
    Z = np.array([np.array([uniform(hmin,hmax) for i in range (taille)]) for i in range (taille)])
    return(np.sort(Z, 0))

def Z_aleatoire_vallee(taille,hmin,hmax):
    z =[]
    for i in range (taille):
         P = [uniform(hmin,hmax) for i in range (taille)]
         z += (sorted(P[0:int(len(P)/2)],reverse=True)+sorted(P[int(len(P)/2):taille]))
    return(np.array(z).reshape((taille,taille)))


def aleat_surface(taille,hmin,hmax):
    Xale = [i for i in range (taille)]
    Yale = [i for i in range (taille)]
    Zale = Z_aleatoire_vallee(taille,hmin,hmax)
    #Zale[Zale > 0.8*hmax]=uniform(0.3*hmax,0.9*hmax)
    ax= plt.axes(projection='3d')
    Xale,Yale=np.meshgrid(Xale,Yale)
    ax.plot_surface(Xale,Yale,Zale,cmap='terrain')
    ax.set_zlim(hmin,hmax+2)
    ax.set_xlabel('x position (mètres)')
    ax.set_ylabel('y position (mètres)')
    ax.set_zlabel('hauteur (mètres)')
    plt.show()
