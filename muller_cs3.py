import random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp 
import sys
import pandas as pd#for readcsv

#smoothing kernels defined in Müller and their gradients
H=0.09
M_PI=3.1415926535897932385
POLY6 = 315.0/(64.0*M_PI*math.pow(H, 9.0));
#POLY6 = 315.0/(65.0*M_PI*math.pow(H, 9.0));
SPIKY_GRAD = -45.0/(M_PI*math.pow(H, 6.0));
VISC_LAP = 45.0/(M_PI*math.pow(H, 6.0));

class Point:
  #def __init__(self): #Constructor, replace by initialization
  #  self.x=0
  #  self.y=0
  def __init__(self,_x=0,_y=0): #Constructor
    self.x=_x
    self.y=_y
    self.rho=0;
    self.label=0

  def print(self):
        print("Point at ",self.x, self.y)
  def string(self):
    return  "Point at {0} {1}".format(self.x, self.y)
  
def norm(p1,p2):
  return  math.sqrt( (p1.x-p2.x)**2+(p1.y-p2.y)**2 )

def filterAround(Points,cx,cy,r):
  F=[]
  for p in Points:
    if ((cx-p.x)**2+(cy-p.y)**2 - r**2)<0:
      F.append(p)
  return F #points

def neighbors(p_r,cloud):
  result=[]
  for p_j in cloud:
     if norm(p_r,p_j)< H:
        result.append(p_j)
  return result


def readcsv(filename):
  #df = pd.read_csv(filename, header=None, sep = ',')
  df = pd.read_csv(filename, sep = ',')
  size=df.shape #(rows,cols)
  df = df[["Points:0","Points:1","Points:2"]]
  M=df.values #.as_matrix()
  P=M[:,[0,2]] # [:,[colx,coly,...]], start in zero [0,1,...]
  return P
def sample(X,n):
   sh=X.shape
   print(sh)
   ies = np.random.randint(sh[0],size=n)
   R=[]
   for i in ies:
     #print(X[i])
     R.append(X[i]) 
   return np.array(R)

def cs(p,A):
    Ns=neighbors(p,A) #defined by H
    rho=0.0
    for p_j in Ns:
      rho+=POLY6*math.pow(H**2-norm(p,p_j)**2,3)
    return rho

def derivate(r):
  return -3*(H**4)*r+3*(H**2)*(r**3)-r**5

def gradcs(p,A,var=0): #0 means x, 1 means y
    Ns=neighbors(p,A) #defined by H
    rhox=0.0
    rhoy=0.0
    for p_j in Ns:
      rhox+=POLY6*derivate(p.x-p_j.x)
      rhoy+=POLY6*derivate(p.y-p_j.y)
    return rhox,rhoy

def pltgrad(A):
 dx = 0.04
 dy = 0.04
 xr = np.arange(-0.1, 1, dx)
 yr = np.arange(-0.1, 1.8, dy)

 # Create grid corresponding to xr and yr arrays
 xx, yy = np.meshgrid (xr, yr, indexing = 'ij')
 xs=xx.shape[0]
 ys=xx.shape[1]
 zz=np.zeros((xs,ys))
 print("grad..")
 for i in range(0,xs):
   for j in range(0,ys):
     p=Point(xx[i,j],yy[i,j])
     zz[i,j] = cs(p,A)/160000

 gradx, grady = np.gradient (zz, dx, dy)
 gradz=np.sqrt(gradx**2+grady**2)
 gradx= gradx/gradz
 grady= grady/gradz

 n = 55
 l = np.array([0.05,0.1, 0.2, 0.3, 0.4])

 plt.contourf(xx, yy, zz, n,cmap=cm.cividis) #gist_earth/cubehelix/Greys
 plt.contour(xx, yy, zz, levels = l, colors = 'r', linewidths = 1, linestyles = 'solid')
 plt.quiver(xx, yy, gradx , grady)
 plt.show()
  
def main(filename):
  X=readcsv(filename) #readcsv 11236
  X=sample(X,3000) 
  fig, ax = plt.subplots(figsize=(4,5))
  i=0
  A=[]
  for x in X: #0..9
    p = Point(x[0],x[1])
    p.label=i
    A.append(p)
    i+=1
  A=filterAround(A,0,0,1.5) 

  for p in A:
   csp=cs(p,A)
   p.rho=csp/160000.0 #for normalize    

  xs=[p.x for p in A]
  ys=[p.y for p in A]
  vs=[p.rho for p in A]
#  vmax=max(vs)
  colors=[(p.rho) for p in A]
  sc=plt.scatter(xs,ys,c=colors,cmap=cm.bone,s=10) #cm.gray/ocean/Blues
  plt.colorbar(sc)
  plt.axis([0, 1.6, 0, 2])
  plt.figure()
  pltgrad(A)
  plt.show()
if __name__ == "__main__":

  filename='/home/DRIVE/Pesquisa2018/fluids/marco/pdata.{0}.csv'.format(sys.argv[1])
  main(filename)
