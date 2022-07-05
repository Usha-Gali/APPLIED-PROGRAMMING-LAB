import scipy.signal as sp
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

#plotting function 
def plot_function(x,y,xlabel,ylabel,title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x,y)
    plt.show()

#defining transfer functions for given frequency and decay
def transfer_function(freq,decay):
    k= np.polymul([1.0,0,2.25],[1,-2*decay,freq*freq + decay*decay])
    H= sp.lti([1,-1*decay],k)
    return H

t,x = sp.impulse(transfer_function(1.5,-0.5),None,np.linspace(0,50,5001)) #computing impulse response of the transfer function
plot_function(t,x,"t","x","Forced Damping Oscillator with decay = 0.5")

t,x = sp.impulse(transfer_function(1.5,-0.05),None,np.linspace(0,50,5001)) #computing impulse response of the transfer function
plot_function(t,x,"t","x","Forced Damping Oscillator with decay = 0.05")

frequency = np.linspace(1.4,1.6,5)
for freq in frequency:
    H = sp.lti([1],[1,0,2.25]) #defining a transfer function
    t = np.linspace(0,150,5001)
    F = np.cos(freq*t)*np.exp(-0.05*t)*(t>0) 
    t,y,svec = sp.lsim(H,F,t) #simulates y=convolution of H and F 
    plt.title("Forced Damping Oscillator with %.2f frequency" % freq)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.plot(t,y)
    plt.show()

#solve for X in coupling equation
X = sp.lti([1,0,2],[1,0,3,0])
t,x = sp.impulse(X,None,np.linspace(0,20,5001)) #computing impulse response of the transfer function
plot_function(t,x,"t","x","Coupled Oscilations: X",)

#solve for Y in coupling equation
Y = sp.lti([2],[1,0,3,0])
t,y = sp.impulse(Y,None,np.linspace(0,20,5001)) #computing impulse response of the transfer function
plot_function(t,y,"t","y","Coupled Oscilations: Y")

def RLC(time,R,L,C,f):    
    H = sp.lti([1],[L*C,R*C,1])
    w,S,phi = H.bode()
    plt.subplot(2,1,1)
    plt.title("Magnitude response")
    plt.xlabel("w")
    plt.ylabel(r'$|H(s)|$')
    plt.semilogx(w,S)
    plt.show()
    plt.subplot(2,1,2)
    plt.title("Phase response")
    plt.xlabel("w")
    plt.ylabel(r'$\angle(H(s))$')
    plt.semilogx(w,phi)
    plt.show()
    return sp.lsim(H,f,time)  #simulates y=convolution of H and F 

t=np.linspace(0,30e-6,1000)
R=100
L=1e-6
C=1e-6
function= np.cos(1000*t) - np.cos(1e6*t)
t,y,svec = RLC(t,R,L,C,function) #finding magnitude and phase response and output voltage
plot_function(t,y,"t",r'$v_{o}(t)$',"Output of RLC for t<30u sec")

t=np.linspace(0,10e-3,100000)
#computing the output voltage v0(t) by defining the transfer function as a system and obtaining the output using signal.lsim
function= np.cos(1000*t) - np.cos(1e6*t)
H = sp.lti([1],[L*C,R*C,1])
w,S,phi = H.bode()
t,y,svec = sp.lsim(H,function,t)
plot_function(t,y,"t",r'$v_{o}(t)$',"Output of RLC for t<10m sec")




