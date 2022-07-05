import numpy as np 
import scipy.signal as sp 
import sympy
sympy.init_session
from sympy import *
import pylab as p

def plot_graph(x, y, xlabel, ylabel):
    p.plot(x,y,'-r',label=r'$V_{o}$')
    p.xlabel(xlabel)
    p.ylabel(ylabel)
    p.legend(loc ='upper right')
    p.grid()
    p.savefig('fig{}.png'.format(fignum[0]))
    fignum[0] += 1
    p.close()

def Highpass_filter(R1,R3,C1,C2,G,Vi): #High pass filter 
    s = sympy.symbols('s')
    A = sympy.Matrix([[0,0,1,-1/G],[-1/(1+1/(s*R3*C2)),1,0,0],[0,-G,G,1],[-s*C1-s*C2-1/R1,s*C2,0,1/R1]])
    b = sympy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)

def Lowpass_filter(R1, R2, C1, C2, G, Vi):#Lowpass filter 
    s = sympy.symbols('s')
    A = sympy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sympy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

def sympy_to_lti(Y, s=sympy.symbols('s')): #converts Sympy transfer function polynomial to Scipy LTI 
    #returns the expressions
    num, den = sympy.simplify(Y).as_numer_denom()
    n,d = sympy.poly(num), sympy.poly(den)
    #return the coefficients
    num,den = n.all_coeffs(),d.all_coeffs()
    #convert to floats
    l_num, l_den = [float(f) for f in num] ,[float(f) for f in den]  
    return (l_num, l_den)


fignum = [0]

#step response
A,b,V=Lowpass_filter(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]
H = sympy_to_lti(Vo)
t = np.linspace(0,0.001,1000)
Vo = sp.step(H,T=t)
plot_graph(Vo[0], Vo[1],r't$\rightarrow$',r'$V_{o}\rightarrow$')

#response for mixed frequency sinusoid
t = np.linspace(0,0.01,100000)
Vi = np.multiply((np.sin(2000*np.pi*t)+np.cos(2000000*np.pi*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)

p.plot(Vo[0],Vi,label=r'$V_{in}$')
plot_graph(Vo[0],Vo[1],r't$\rightarrow$',r'$V\rightarrow$')

#magnitude response of high pass filter
A,b,V = Highpass_filter(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]
H = sympy_to_lti(Vo)
w = p.logspace(0,8,801)
ss = 1j*w
hf = sympy.lambdify(sympy.symbols('s'),Vo,'numpy')
v = hf(ss)
p.loglog(w,abs(v),lw=2)
p.xlabel(r'$w\rightarrow$')
p.ylabel(r'$|H(jw)|\rightarrow$')
p.grid(True)
p.savefig('fig{}.png'.format(fignum[0]))
fignum[0] += 1
p.close()

#response of circuit to a damped sinusoids
t = np.linspace(0,10,1000)
Vi = np.multiply(np.multiply(np.exp(-0.5*t),np.sin(2*np.pi*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)
p.plot(Vo[0],Vi,label=r'$V_{in}$')
plot_graph(Vo[0],Vo[1],r't$\rightarrow$',r'$V\rightarrow$')

t = np.linspace(0,0.0001,10000)
Vi = np.multiply(np.multiply(np.exp(-0.5*t),np.sin(2*np.pi*200000*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)
p.plot(Vo[0],Vi,label=r'$V_{in}$')
plot_graph(Vo[0],Vo[1],r't$\rightarrow$',r'$V\rightarrow$')

# step response 
t = np.linspace(0,0.001,1000)
Vo = sp.step(H,T=t)
plot_graph(Vo[0],Vo[1],r't$\rightarrow$',r'$V_{o}\rightarrow$')