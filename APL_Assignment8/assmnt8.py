from itertools import count
from pylab import *
import numpy as np

def f0(x):
    return (1+0.1*cos(x))*cos(10*x)
def f1(x):
    return (sin(x))**3
def f2(x):
    return (cos(x))**3

def spectrum(f,xlimit,TITLE1,Xlabel,Ylabel1,Ylabel2):
    x = linspace(-4*np.pi,4*np.pi,513)[:-1]
    y = f(x)
    Y = fftshift(fft(y))/512.0
    w = linspace(-64,64,513);w = w[:-1]
    figure()
    subplot(2,1,1)
    grid(True)
    plot(w,abs(Y),lw=2)
    xlim([-xlimit,xlimit])
    title(TITLE1)
    xlabel(Xlabel)
    ylabel(Ylabel1)
    subplot(2,1,2)
    plot(w,angle(Y),'ro',lw=2)
    xlim([-xlimit,xlimit])
    xlabel(Xlabel)
    ylabel(Ylabel2)
    grid(True)
    show()


#given example 1
x=rand(100)
X=fft(x)
y=ifft(X)
c_[x,y]
print('absolute maaximum error= ',abs(x-y).max())

x=linspace(0,2*pi,128)
y=sin(5*x)
Y=fft(y)
figure()
subplot(2,1,1)
plot(abs(Y),lw=2)
ylabel(r"$|Y|$")
title(r"Spectrum of $\sin(5t)$ without phase shift")
grid(True)
subplot(2,1,2)
plot(unwrap(angle(Y)),lw=2)
ylabel(r"Phase of $Y$")
xlabel(r"$k$")
grid(True)
show()

#given example 2
x=linspace(0,2*pi,129);x=x[:-1]
y=sin(5*x)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$")
title(r"Spectrum of $\sin(5t)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
ii=where(abs(Y)>1e-3)
plot(w[ii],angle(Y[ii]),'go',lw=2)
xlim([-10,10])
ylabel(r"Phase of $Y$")
xlabel(r"$k$")
grid(True)
show()

#given example 3
t=linspace(0,2*pi,129);t=t[:-1]
y=(1+0.1*cos(t))*cos(10*t)
Y=fftshift(fft(y))/128.0
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$")
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"Phase of $Y$")
xlabel(r"$\omega$")
grid(True)
show()

#given example 4
spectrum(f0,15,r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$",r"$\omega$",r"$|Y|$",r"Phase of $Y$")

#generating spectrums of sin^3(t) and cos^3(t)
spectrum(f1,15,r"Spectrum of $sin^3(t)$",r"$\omega$",r"$|Y|$",r"Phase of $Y$")
spectrum(f2,15,r"Spectrum of $cos^3(t)$",r"$\omega$",r"$|Y|$",r"Phase of $Y$")

#generating spectrum of cos(20t + 5cos(t))
t = linspace(-4*np.pi,4*np.pi,513)[:-1]
y = cos(20*t + 5*cos(t))
Y = fftshift(fft(y))/512.0
w = linspace(-64,64,513);w = w[:-1]
figure()
subplot(2,1,1)
grid(True)
plot(w,abs(Y),lw=2)
xlim([-40,40])
title(r"Spectrum of $cos(20t + 5cos(t))$")
xlabel(r"$\omega$")
ylabel(r"$|Y|$")
subplot(2,1,2)
plot(w,angle(Y),lw=2)
ii = where(abs(Y)> 0.001)   #plotting phase points where the magnitude is greater than 0.001
plot(w[ii],angle(Y[ii]),'ro')
xlim([-40,40])
xlabel(r"$\omega$")
ylabel(r"$|Y|$")
grid(True)
show()

#generating spectrum of exp(-t^2/2)
T = 2*pi
N = 128
count = 0
Tolerance = 1e-15
error = Tolerance + 1

# calculating the DFT and error between the calculated and actual value.
while error>Tolerance:

	t = linspace(-T/2,T/2,N+1)[:-1]
	w = N/T * linspace(-pi,pi,N+1)[:-1] 
	y = exp(-0.5*t**2)
	count = count + 1
	Y = fftshift(fft(y))*T/(2*pi*N)
	actual_Y = (1/sqrt(2*pi))*exp(-0.5*w**2)
	error = mean(abs(abs(Y)-actual_Y))

	if error < Tolerance:
		break
	
	T = T*2
	N = N*2

print(" Error: %g \n " % (error))
print(" Best value for T: %g*pi \n Best value for N: %g"%(T/pi,N))

# Magnitude plot for the most accurate DFT of the Gaussian. 
figure()
xlim([-10,10])
plot(w,abs(Y))
xlabel(r"$\omega$")
ylabel(r"$|Y|$")
title(r"Spectrum of a Gaussian function")
grid(True)
show()
