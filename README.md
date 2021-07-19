# N-Body-Problem
Numerical simulation of the classic n-body-problem using python and julia. 

For some background information (German only) see: 
http://www.tat.physik.uni-tuebingen.de/~kley/lehre/cp-prakt/projekte/projekt-speith.pdf

Input data can be found on this website:
http://www.pit.physik.uni-tuebingen.de/~speith/Projekt1/

## Example problem solved with...

### Euler method

![Euler method gif](https://github.com/JoshuaSimon/N-Body-Problem/blob/master/Julia/R_euler_method.gif)

### Verlet algorithm

![Verlet algorithm gif](https://github.com/JoshuaSimon/N-Body-Problem/blob/master/Julia/R_verlet.gif)

### Euler-Cormer method

![Euler-Cormer method gif](https://github.com/JoshuaSimon/N-Body-Problem/blob/master/Julia/R_euler_cormer.gif)

## Simple solar system simulation

![Solar system gif](https://github.com/JoshuaSimon/N-Body-Problem/blob/master/Julia/Solar_verlet.gif)

I couldn't find any data for the relative coordinates of the bodies, so I went with placing them on one axis in the distance of their radii and just shot them straight up according to v=√(GM/r).
