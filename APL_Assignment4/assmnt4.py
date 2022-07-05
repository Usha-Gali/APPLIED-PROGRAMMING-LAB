import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math

#defining python functions for exp(x) and cos(cos(x))
def exp(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))

#Plotting exp(x) on semilogy graph and cos(cos(x)) functions
def exp_plot(lowerlimit,upperlimit, n):
    x = np.linspace(lowerlimit,upperlimit,n)
    plt.grid()
    plt.semilogy(x,exp(x))
    plt.title(r'$exp^x$ on a semilogy plot')
    plt.xlabel('$x$')
    plt.ylabel(r'$log(exp^x)$')
    plt.savefig('exp_plot.png')
    plt.close()

def coscos_plot(lowerlimit,upperlimit, n):  
    x = np.linspace(lowerlimit,upperlimit,n)
    plt.grid()
    plt.title(r'Plot of $cos(cos(x))$')
    plt.xlabel('$x$')
    plt.ylabel(r'$cos(cos(x))$')
    plt.plot(x,coscos(x))
    plt.savefig('coscos_plot.png')
    plt.close()

#calculating coefficients for the two functions
def FT(n,function):
    a = np.zeros(n)

    def u(x,k,f):
        u=f(x)*(np.cos(k*x))
        return u/np.pi
    def v(x,k,f):
        v=f(x)*(np.sin(k*x))
        return v/np.pi

#quad integrates the function by taking limits,function and extra arguments
    a[0] = integrate.quad(function,0,2*np.pi)[0]/(2*np.pi)

    for i in range(1,n):
        if(i%2==1):
            a[i] = integrate.quad(u,0,2*np.pi,args=(int(i/2)+1,function))[0]
        else:
            a[i] = integrate.quad(v,0,2*np.pi,args=(int(i/2),function))[0]
    return a

# plotting eFt_coeff and coscosFt_coeffa,coscosFt_coeffb in semilogy and the loglog scale
def plot_semilog_loglog(eFt_coeff,coscosFt_coeff,color = 'ro'):
    eFt_coeff = np.abs(eFt_coeff)
    coscosFt_coeff = np.abs(coscosFt_coeff)
    plt.grid()
    plt.title(r"Coefficients of fourier series of $exp^x$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(eFt_coeff,color)
    plt.savefig('semilog_eFt_coeff.png')
    plt.close()

    plt.grid()
    plt.title(r"Coefficients of fourier series of $exp^x$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(eFt_coeff,color)
    plt.savefig('loglog_eFt_coeff.png')
    plt.close()

    plt.grid()
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(coscosFt_coeff,color)
    plt.savefig('semilog_coscos_coeff.png')
    plt.close()

    plt.grid()
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(coscosFt_coeff,color)
    plt.savefig('loglog_coscos_coeff.png')
    plt.close()

#This function Generates Matrices A and b
def generateA_b(f,x):
    A = np.zeros((400,51))
    A[:,0] = 1 #column1 in A is 1s
    for k in range(1,26):
        A[:,2*k-1]=np.cos(k*x)
        A[:,2*k]=np.sin(k*x)
    return A,f(x)

# plots Fourier coefficients obtained by the two methods in semilogy and the loglog scale
def plot(eFt_coeff,coscosFt_coeff,c_exp,c_coscos,color):
    eFt_coeff = np.abs(eFt_coeff)
    coscosFt_coeff = np.abs(coscosFt_coeff)
    
    plt.grid()
    plt.title(r"Coefficients of fourier series of $exp^x$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(eFt_coeff,'ro')
    plt.semilogy(c_exp,color)
    plt.legend(["true","prediction"])
    plt.savefig('fig1.png')
    plt.close()

    plt.grid()
    plt.title(r"Coefficients of fourier series of $exp^x$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(eFt_coeff,'ro')
    plt.loglog(c_exp,color)
    plt.legend(["true","prediction"])
    plt.savefig('fig2.png')
    plt.close()

    c_exp = np.abs(c_exp)
    c_coscos = np.abs(c_coscos)

    plt.grid()
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(coscosFt_coeff,'ro')
    plt.semilogy(c_coscos,color)
    plt.legend(["true","prediction"])
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a semilogy scale")
    plt.savefig('fig3.png')
    plt.close()

    plt.grid()
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(coscosFt_coeff,'ro')
    plt.loglog(c_coscos,color)
    plt.legend(["true","prediction"])
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a loglog scale")
    plt.savefig('fig4.png')
    plt.close()


#Plotting exp(x) on semilogy graph and cos(cos(x)) functions
exp_plot(-2*np.pi,4*np.pi,1000)
coscos_plot(-2*np.pi,4*np.pi,1000)

#calculating fourier coefficients of both the functions
eFt_coeff = FT(51,exp)
coscosFt_coeff = FT(51,coscos)

# plotting eFt_coeff and coscosFt_coeffa,coscosFt_coeffb in semilogy and the loglog scale
plot_semilog_loglog(eFt_coeff,coscosFt_coeff)


x = np.linspace(0,2*np.pi,401)
x=x[:-1]  #dropping last term to have a proper periodic integral

#Generating A and b matrices for cos(cos(x)),exp(x) functions
A_coscos,b_coscos = generateA_b(coscos,x)
A_exp,b_exp = generateA_b(exp,x)

# solving using lstsq
c_coscos = scipy.linalg.lstsq(A_coscos,b_coscos)[0]
c_exp = scipy.linalg.lstsq(A_exp,b_exp)[0]

#plotting fourier coeff for both functions obtained from both the methods (matrices method in green circles)
plot(eFt_coeff,coscosFt_coeff,c_exp,c_coscos,'go')

#measuring absolute error between two methods 
print("The error in Coefficients of exp^x =",np.amax(np.abs(eFt_coeff - c_exp))) #amax() returns the maximum value in that array
print("The error in Coefficients of cos(cos(x)) =",np.amax(np.abs(coscosFt_coeff - c_coscos)))

#reshape() shapes array without changing data 
c_exp = np.reshape(c_exp,(51,1))

f_values = np.matmul(A_exp,c_exp)
#plotting results
x = np.linspace(0,2*np.pi,400)
plt.grid()
plt.title(r"Plot of $exp^x$")
plt.xlabel('x')
plt.ylabel(r'$log(exp^x)$')
t = np.linspace(-2*np.pi,4*np.pi,1000)
plt.semilogy(t,exp(t))
plt.semilogy(x,f_values,'go')
plt.legend(["true","prediction"])
plt.savefig('fig5.png')
plt.close()

c_coscos = np.reshape(c_coscos,(51,1))

f_values = np.matmul(A_coscos,c_coscos)
#plotting results
x = np.linspace(0,2*np.pi,400)
plt.grid()
plt.title(r"Plot of $cos(cos(x))$")
plt.xlabel('x')
plt.ylabel(r'$cos(cos(x))$')
t = np.linspace(-2*np.pi,4*np.pi,1000)
plt.plot(x,f_values,'ro')
plt.plot(t,coscos(t))
plt.legend(["true","prediction"])
plt.savefig('fig6.png')
plt.close()

