from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import *

## 2D

levels = np.linspace(0., 10., 100)
plt.contourf(X, Y, Z, levels,cmap='terrain')#60,cmap = mpl.cm.jet)
plt.colorbar()
plt.show()

## 3D

X = [i for i in range (25)]
Y = [i for i in range (25)]
Z = np.asarray(Z1)
ax= plt.axes(projection='3d')
X,Y=np.meshgrid(X,Y)
ax.plot_surface(X,Y,Z,cmap='terrain',linewidth=0, antialiased=False)
#plt.contourf(X, Y, Z, 20, cmap = "plasma")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Hauteur')
#plt.colorbar()
plt.show()


## 3D interpolated ??

from scipy.interpolate import interp1d
import scipy.interpolate

x = [i for i in range (25)]
y = [i for i in range (25)]
z = Z
ax= plt.axes(projection='3d')
for i in range (25):
    f=interp1d(x,z[i],kind='cubic')
    xnew = np.linspace(0, 24, 100)
    result = f(xnew)
    plt.plot( xnew,result, 'b-')
plt.show()





##

z1=[0,0,0,1,2,2,2,3,4,5,5,5,5,5,4,4,5,4,4,4,4,3,3,2,2]
z2=[0,1,1,2,2,3,3,3,4,5,6,6,6,5,5,5,6,5,5,5,4,4,3,3,3]
z3=[1,1,2,2,3,3,3,4,4,5,6,6,6,6,6,6,6,6,6,5,5,4,4,3,3]
z4=[1,2,2,3,3,3,4,4,5,6,6,6,7,7,7,6,6,6,6,5,5,5,4,4,3]
z5=[2,2,3,3,4,4,4,4,5,6,6,7,7,7,7,7,7,7,6,6,5,5,5,4,4]
z6=[2,2,3,4,4,4,5,5,6,6,7,7,7,7,7,7,7,7,6,6,6,6,5,5,4]
z7=[2,3,4,4,4,5,5,6,6,7,7,7,8,8,8,8,7,7,7,7,7,6,5,5,4]
z8=[3,3,4,4,5,5,6,6,7,7,7,8,8,9,8,8,8,7,7,7,7,6,6,5,5]
z9=[3,4,4,4,5,6,6,7,7,8,8,9,9,9,9,8,8,8,7,7,7,6,6,6,5]
z10=[3,4,4,5,5,6,6,7,7,8,8,9,10,10,9,8,8,7,7,7,7,6,6,6,5]
z11=[3,4,4,5,6,6,7,7,8,8,9,10,10,9,9,8,7,7,7,7,6,6,6,6,5]
z12=[3,4,4,5,6,6,7,7,8,8,9,9,9,9,8,7,7,7,7,6,6,6,5,5,4]
z13=[3,4,4,5,6,6,7,7,8,8,8,9,9,9,8,7,7,7,7,6,6,5,5,4,4]
z14=[3,4,4,5,5,6,6,7,7,8,8,8,8,8,8,7,7,7,7,6,6,5,5,4,3]
z15=[3,3,4,4,5,6,6,7,7,7,7,7,8,8,8,7,7,7,7,6,6,5,5,4,4]
z16=[3,3,4,4,5,6,6,6,6,7,7,7,7,7,7,7,7,7,6,6,6,5,5,5,4]
z17=[3,3,4,4,5,6,6,6,6,6,6,7,7,7,7,7,6,6,6,6,5,5,5,4,3]
z18=[2,3,4,4,5,5,5,6,6,6,6,6,7,7,6,6,6,6,5,5,5,5,4,4,3]
z19=[2,3,4,4,4,5,5,5,5,4,5,6,6,6,6,6,5,5,5,4,4,4,4,3,3]
z20=[2,3,4,4,4,4,4,4,3,3,5,6,6,6,6,5,5,4,4,4,4,4,3,3,2]
z21=[2,3,3,3,3,3,3,3,2,4,5,6,6,6,5,4,4,4,3,3,3,3,2,2,2]
z22=[2,2,3,3,3,3,3,2,2,4,4,5,5,5,5,4,4,3,3,2,2,2,2,1,1]
z23=[1,2,2,2,2,2,2,1,2,4,4,4,5,5,4,4,3,3,2,2,1,1,1,1,1]
z24=[1,1,1,1,1,1,1,1,2,4,4,4,4,4,4,3,3,2,1,1,1,1,0,0,0]
z25=[0,0,0,0,0,0,0,0,1,2,3,3,3,3,3,2,2,1,1,1,0,0,0,0,0]

Z1 = [z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25]




## New data

import pandas as pd

#importation des files
data = pd.read_excel('/Users/antoninlefevre/Downloads/TIPE mp/TIPE/surface/donnees_bathy.xls')


#on choisi laquelle étudier
bathy = data.dropna(axis=0)

#manipulation
bathy.columns=['Xposition','Yposition','Zposition']
n=int(np.floor(np.sqrt(len(bathy['Xposition']))))
m=int(n**2)
max = max(bathy['Zposition'][0:m])
min = min(bathy['Zposition'][0:m])


X1 = bathy['Xposition'][0:n].sort_values(axis = 0, ascending = False)
Y1 = bathy['Yposition'][0:n].sort_values(axis = 0, ascending = False)
Z1 = bathy['Zposition'][0:m].to_numpy().reshape((n,n))

levels = np.linspace(min,max,n)
plt.contourf(X1, Y1, Z1, levels,cmap='terrain')#60,cmap = mpl.cm.jet)
plt.colorbar()
plt.show()

ax=plt.axes(projection='3d')
X1,Y1=np.meshgrid(X1,Y1)
ax.plot_surface(X1,Y1,Z1,cmap='terrain',linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Hauteur')
plt.show()

## surface aléatoire nxn

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


## PolynomialFeatures

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler

def aleatoire_surface(n):
    X1 = np.random.randint(2,7,[n,1])
    array_plot = PolynomialFeatures(n-1).fit_transform(X1)
    X = [i for i in range (n)]
    Y = [i for i in range (n)]
    Z = RobustScaler().fit_transform(array_plot)
    ax= plt.axes(projection='3d')
    X,Y=np.meshgrid(X,Y)
    ax.plot_surface(X,Y,Z,cmap='terrain')
    plt.show()











