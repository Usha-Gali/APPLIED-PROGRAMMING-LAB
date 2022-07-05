from pylab import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Given parameters of the problem
l = 0.5  
c = 2.9979e8  
a = 0.01 
N = 4  
Im = 1.0
mu0 = 4e-7 * (np.pi)

lamda = l * 4.0
frequency = c / lamda
k = 2 * np.pi / lamda
dz = l / N

# Question 1
z = np.linspace(-l, l, 2*N + 1) #defining points along antenna as array z
I = np.zeros(2*N + 1) ##creating zero array to store I values
I[0], I[N], I[2*N] = 0, Im, 0  #Given endcurrents are 0 and current in the middle is Im

#constructing current vector I corresponding to z
I[0:N] = Im * np.sin(k * (l + z[0:N]))
I[N : 2*N + 1] = Im * np.sin(k * (l - z[N : 2*N + 1]))

u = [j for j in range(1, 2*N)]
u.pop(N - 1)
u = np.array(u, dtype=int)
print('Vector z: ', z.round(2))
print('Vector u: ', u.round(2))
#creating current vector J corresponding to u
J = I[u]

#printing all vectors
print('Vector I: ', I.round(2))
print('Vector J: ', J.round(2))
print('')

# Question 2
#creating a function to compute matrix M and returning it
def M(n,r):
    M = np.identity(2*n - 2)
    M = (1 / (2*np.pi*r)) * M
    return M

M = M(N,a)
print('Matrix M: ', M.round(2))

# Question 3
#Rz and Ru which are the distances from observer and from source

Rz = np.zeros((2*N + 1, 2*N + 1))
for j in range(0, 2*N + 1):
    for i in range(0, 2*N + 1):
        Rz[j][i] = np.sqrt(a*a + (z[j] - z[i])*(z[j] - z[i]))

#Ru is the vector of distances to unknown currents
Ru = np.zeros((2*N - 2, 2*N - 2))
for j in range(0, 2*N - 2):
    for i in range(0, 2*N - 2):
        Ru[j][i] = np.sqrt(a*a + (u[j] * dz - u[i] * dz) * (u[j] * dz - u[i] * dz))

RiN = Rz[N]
RiN = np.delete(RiN, [0, N, 2*N], 0)

print('Vector Rz: ', Rz.round(2))
print('Vector Ru: ', Ru.round(2))

P = np.zeros((2*N - 2, 2*N - 2), dtype=complex)
for j in range(2*N - 2):
    for i in range(2 * N - 2):
        P[j][i] = (mu0 / (4.0 * np.pi)) * (np.exp(-1j * k * Ru[j][i])) * dz / Ru[j][i]

#PB is the contribution to the vector potential due to current IN
PB = (mu0 / (4*np.pi)) * (np.exp(-1j*k*RiN)) * dz / RiN

print('Matrix P: ',(P*1e8).round(2))
print('Matrix PB: ',(PB*1e8).round(2))
print('')

#Question 4
#creating matrices Qij and QB
Q = np.zeros((2*N - 2, 2*N - 2), dtype=complex)
for i in range(2*N - 2):
    for j in range(2*N - 2):
        Q[i][j] = -P[i][j] * (a / mu0) * ((-1j * k / Ru[i][j]) - (1 / pow(Ru[i][j], 2)))

QB = -PB * (a / mu0) * ((-1j * k / RiN) - (1 / RiN ** 2))

print('Matrix Q: ', Q.round(2))
print('Vector QB: ', QB.round(2))
print('')

#Question 5

calculated_J = np.dot(np.linalg.inv(M - Q), QB) 
calculated_I = np.zeros(2*N+1, dtype = complex) #creating zero array to store I values
#applying given boundary conditions -----> (zero at i=0, i=2N, and Im at i=N)
calculated_I[1:N] = calculated_J[0:N-1]
calculated_I[N+1: 2*N] = calculated_J[N-1:2*N-1]
calculated_I[N] = Im  

#plotting calculated currents and approximate Current expression
plt.plot(z, I)
plt.plot(z, calculated_I)
plt.title('Plot of calculated current and approximate current expression')
plt.xlabel('Element index',size=15)
plt.ylabel('Current',size=15)
plt.legend(['Approximate current expression','Calculated current'])
plt.grid(True)
plt.show()