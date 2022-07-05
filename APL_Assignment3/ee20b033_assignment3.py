import matplotlib as plt
from pylab import *
import scipy.special as sp
import numpy as np 
import pylab

def loading_file(filename):
    data = [] #creating empty list
    data = np.loadtxt(filename,dtype=float) 
    t  = np.array(data[:,0]) #first column is time
    y_data = np.array(data)[:,1:] #all remaining columns are data
    return t,y_data

#creating a python function that computes g(t,A,B)
def g(t,A,B):
    k= A*sp.jn(2,t) + B*t
    return k

def plot_addlabels(t,y_data,A,B): #plotting the graph and adding labels
    pylab.grid() 
    pylab.title('Q4:Data to be fitted to theory') #adding title
    for i in range(9):
        pylab.plot(t, y_data[:,i], label=r'$\sigma=%.3f$'%stdev[i])
    pylab.plot(t, g(t,A,B), color="black", label='True Value')
    pylab.legend() #adding legend
    #adding x and y labels
    pylab.xlabel(r't$\rightarrow$', fontsize=20)
    pylab.ylabel(r'f(t)+noise$\rightarrow$',fontsize=20)
    pylab.savefig('plot_addlabels.png') #saving image
    pylab.close() #closing the image

def plot_errorbar(t,y_data,A,B):
    data = y_data[:,0] #extracting first columns of data
    pylab.grid()
    pylab.title('Q5:Data points for stdev=0.10 along with exact function') #adding title
    pylab.errorbar(t[::5], data[::5], stdev[0], fmt='ro' , label='Errorbar') #plotting every 5th data item
    pylab.plot(t, g(t,A,B), 'black' ,label='f(t)') #plotting g(t) function 
    pylab.legend() #adding legend
    #adding x,y labels
    pylab.xlabel(r't$\rightarrow$',fontsize=15)
    pylab.ylabel(r'f(t)$\rightarrow$',fontsize=15)
    pylab.savefig('plot_errorbar.png') #saving figure
    pylab.close() #closing the image

def contour_plot(e, A, B):
    plot = pylab.contour(A,B,e[:,:,0],20) #contour plots contour lines
    pylab.grid()
    #adding x,y labels
    pylab.xlabel(r'A$\rightarrow$')
    pylab.ylabel(r'B$\rightarrow$')
    pylab.title("Q8: contour plot of mean squared array") #adding title
    pylab.clabel(plot,inline=1,fontsize=10) #adding contour labels
    #np.unravel_index used to convert flat indices to tuple of coordinate arrays
    a = np.unravel_index(np.argmin(e[:,:,0]),e[:,:,0].shape) #argmin() returns minimum element of the array
    pylab.plot(A[a[0]],B[a[1]],'o',markersize=5)  #plotting minimum element
    pylab.annotate('(%0.2f,%0.2f)'%(A[a[0]],B[a[1]]),(A[a[0]],B[a[1]]))
    pylab.savefig('contour_plot.png') #saving image
    pylab.close() #closing the image

def error_function(t,y_data): 
    e = np.zeros((21,21,9)) #initializing e with zeros in it
    A = np.linspace(0,2,21) #linspace returns number spaces evenly with respect to interval
    B = np.linspace(-0.2,0,21) 
    for m in range(9):
        f = y_data[:,m]
        for i in range(21):
            for j in range(21):
                e[i][j][m] = np.sum((f - np.array(g(time,A[i],B[j])))**2)/101
    
    return e,A,B

def plot_error_vs_stdev(erra, errb):
    pylab.grid()
    pylab.plot(stdev,erra,'bo',label='Aerr')
    pylab.plot(stdev,errb,'ro',label='Berr')
    pylab.title('Q10: Variation of error with noise')
    pylab.ylabel(r'Noise standard deviation$\rightarrow$',fontsize=15)
    pylab.xlabel(r'$\Error\rightarrow$',fontsize=15)
    pylab.legend(loc='upper left')
    pylab.savefig('error_vs_sigma_plot.png')
    pylab.close()

def plot_error_vs_stdevlog(erra, errb):
    pylab.grid()
    pylab.loglog(stdev,erra,'bo',label='Aerr')
    pylab.stem(stdev,erra, '-bo') #stem() plots vertical lines at each x positions
    pylab.loglog(stdev,errb,'ro',label='Berr')
    pylab.stem(stdev,errb,'-ro')
    pylab.title('Q11: Variation of error with noise')
    pylab.xlabel(r'$\sigma_{n}\rightarrow$',fontsize=15)
    pylab.ylabel(r'Error$\rightarrow$',fontsize=15)
    pylab.legend(loc='upper right')
    pylab.savefig('error_vs_sigma_plot_log_scale.png')
    pylab.close()

N =101 # no of data points
k=9 # no of sets of data with varying noise
A=1.05
B=-0.105 
# generate the data points and add noise

time, y_data = loading_file("fitting.dat") #loading fitting.dat 
stdev = np.logspace(-1,-3,9) #stdev -->standard deviation for noise

plot_addlabels(time,y_data,A,B) #plotting and add labels

plot_errorbar(time,y_data,A,B) #plotting data with errorbars 

function_column = sp.jn(2,time) #bessel function of first kind of real order and complex argument
M = pylab.c_[function_column,time] #construct M by creating column vector
p = np.array([A,B]) 
mult_matrix = np.matmul(M,p) #multiply M and p matrices
matrix_g = np.array(g(time,A,B))
#checking whether these 2 matrices are equal or not
print("Matrix obtained from both methods is same : ", np.array_equal(mult_matrix,matrix_g))

#calculating mean squared error
error_matrix, A_matrix, B_matrix = error_function(time,y_data)

#plot the contour plot of mean squared error
contour_plot(error_matrix, A_matrix, B_matrix)

#Using Python function lstsq to obtain the best estimate of A and B
estimate = [np.linalg.lstsq(M, y_data[:,i], rcond=None)[0] for i in range(9)]
estimate = np.array(estimate)

#abs returns absolute value
A_error = abs(estimate[:,0]-A) 
B_error = abs(estimate[:,1]-B)

plot_error_vs_stdev(A_error, B_error) 
plot_error_vs_stdevlog(A_error, B_error)