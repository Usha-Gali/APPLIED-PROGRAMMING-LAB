from configparser import InterpolationSyntaxError
import numpy as np
import sys
import scipy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

#plotting contour plot of potential
def plot_contour(X,Y,phi,ii):
    plt.title("Contour plot of potential")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.contourf(X,Y,phi)
    plt.colorbar()
    plt.savefig('plot_contour.png')
    plt.close()

def update_phi(phi,phiold):
    #taking average of all neighbours(right,left neighbours and top,bottom neighbours)
    phi[1:-1,1:-1]=0.25*(phiold[1:-1,0:-2]+ phiold[1:-1,2:]+ phiold[0:-2,1:-1] + phiold[2:,1:-1])
    return phi

def boundary(phi,ii):
    phi[0,1:-1]=phi[1,1:-1] #Top boundary
    phi[Ny-1,1:-1]=0 #At bottom edge(ground)
    phi[1:-1,0]=phi[1:-1,1] #Left boundary
    phi[1:-1,Nx-1]=phi[1:-1,Nx-2] #Right boundary
    phi[ii]=1.0
    return phi

def plot_semilog_errors(Niter,errors):
    plt.title("Errors on a semilog plot")
    plt.xlabel("Number of iterations")
    plt.ylabel("Errors")
    plt.semilogy(range(Niter),errors)
    plt.savefig('plot_semilog_errors.png')
    plt.close()

def plot_loglog_errors(Niter,errors):
    plt.title("Errors on a loglog plot")
    plt.xlabel("Number of iterations")
    plt.ylabel("Errors")
    plt.loglog((np.asarray(range(Niter))+1),errors)
    plt.loglog((np.asarray(range(Niter))+1)[::50],errors[::50],'ro')
    plt.legend(["Real","Every 50th value"])
    plt.savefig('plot_loglog_errors.png')
    plt.close()

def get_fit(errors,Niter,lastn=0):
    log_err = np.log(errors)[-lastn:]
    X = np.vstack([(np.arange(Niter)+1)[-lastn:],np.ones(log_err.shape)]).T
    log_err = np.reshape(log_err,(1,log_err.shape[0])).T
    return np.linalg.lstsq(X, log_err, rcond=-1)[0]

def plot_error(errors,Niter,a,a_,b,b_):
    #plotting fits on loglog scale
    plt.title("Best fit for error on a loglog scale")
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    x = np.asarray(range(Niter))+1
    plt.loglog(x,errors)
    plt.loglog(x[::100],np.exp(a+b*np.asarray(range(Niter)))[::100],'ro')
    plt.loglog(x[::100],np.exp(a_+b_*np.asarray(range(Niter)))[::100],'go')
    plt.legend(["Errors","fit1","fit2"])
    plt.savefig('plot_error_loglog.png')
    plt.close()
    
    #plotting fits on semilog scale
    plt.title("Best fit for error on a semilog scale")
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.semilogy(x,errors)
    plt.semilogy(x[::100],np.exp(a+b*np.asarray(range(Niter)))[::100],'ro')
    plt.semilogy(x[::100],np.exp(a_+b_*np.asarray(range(Niter)))[::100],'go')
    plt.legend(["errors","fit1","fit2"])
    plt.savefig('plot_error_semilog.png')
    plt.close()

#finding net error
def net_error(a,b,Niter):
    Error = -a/b*np.exp(b*(Niter+0.5))
    return Error

#plotting cumulative error values on loglog scale
def plot_cumulative_error(iterations,a_,b_):
    plt.grid()
    plt.title(r'Plot of Cumulative Error values on a loglog scale')
    plt.xlabel("Iterations")
    plt.ylabel("Net maximum error")
    plt.loglog(iterations,np.abs(net_error(a_,b_,iterations)),'ro')
    plt.savefig('plot_cumulative_error.png')
    plt.close()

#Plotting 2d contour of final potential
def plot_2d_contour(ii,Nx,Ny,Y,X,phi):
    plt.title("2D Contour plot of potential")
    plt.xlabel("X")
    plt.ylabel("Y")
    x_c,y_c=ii
    plt.plot((x_c-Nx/2)/Nx,(y_c-Ny/2)/Ny,'ro')
    plt.contourf(Y,X[::-1],phi)
    plt.colorbar()
    plt.savefig('plot_2d_contour.png')
    plt.close()

#boundary conditions
def temper(phi,ii):
    phi[:,0]=phi[:,1] # Left Boundary
    phi[:,Nx-1]=phi[:,Nx-2] # Right Boundary
    phi[0,:]=phi[1,:] # Top Boundary
    phi[Ny-1,:]=300.0 #At bottom edge(ground)
    phi[ii]=300.0
    return phi

#laplaces equation
def tempdef(temp,tempold,Jx,Jy):
    temp[1:-1,1:-1]=0.25*(tempold[1:-1,0:-2]+ tempold[1:-1,2:]+ tempold[0:-2,1:-1] + tempold[2:,1:-1]+(Jx)**2 +(Jy)**2)
    return temp

#plotting current density
def plot_current_density(Y,X,Jx,Jy):
    plt.title("Vector plot of current flow")
    plt.quiver(Y[1:-1,1:-1],-X[1:-1,1:-1],-Jx[:,::-1],-Jy)
    x_c,y_c=np.where(X**2+Y**2<(0.35)**2)
    plt.plot((x_c-Nx/2)/Nx,(y_c-Ny/2)/Ny,'ro')
    plt.savefig('plot_current_density.png')
    plt.close()

#plotting 2d contour of final temp
def plot_2dcontour_final(Y,X,temp):
    plt.title("2D Contour plot of temperature")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.contourf(Y,X[::-1],temp)
    plt.colorbar()
    plt.savefig('plot_2dcontour_final.png')
    plt.close()


#Getting user inputs
if(len(sys.argv)==5):
    Nx=int(sys.argv[1])
    Ny=int(sys.argv[2])
    radius=int(sys.argv[3])  
    Niter=int(sys.argv[4])
    print("Using user provided parameters.")
else:
    Nx=25           # size along x
    Ny=25           # size along y
    radius=8        #radius of central lead
    Niter=1500      #number of iterations to perform
    print("Using default values.")

#initializing potential to 0 
phi=np.zeros((Nx,Ny),dtype = float)
x=np.linspace(-0.5,0.5,num=Nx,dtype=float)
y=np.linspace(-0.5,0.5,num=Ny,dtype=float)
Y,X=np.meshgrid(y,x) #meshgrid() returns coordinate matrices from coordinate vectors
R = X*X + Y*Y
ii = np.where(R <= (0.35*0.35)) #where() returns elements which satisfies the condition
phi[ii]=1.0 #setting potential to 1.0 

#plotting contour plot of potential
plot_contour(X,Y,phi,ii)

errors = np.zeros(Niter) 
for k in range(Niter):
    phiold=phi.copy() #copying phi value to phiold
    phi=update_phi(phi,phiold) #updating potential 
    phi=boundary(phi,ii) 
    errors[k]=np.max(np.abs(phi-phiold))

plot_semilog_errors(Niter,errors)
plot_loglog_errors(Niter,errors)

b,a = get_fit(errors,Niter)
b_,a_ = get_fit(errors,Niter,500)
plot_error(errors,Niter,a,a_,b,b_)

iterations=np.arange(100,1501,100)
#plotting cumulative error values on loglog scale
plot_cumulative_error(iterations,a_,b_)

#Plotting 2d contour of final potential
plot_2d_contour(ii,Nx,Ny,Y,X,phi)

#plotting 3d contour of final potential
fig1=plt.figure(4)     # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
plt.title("The 3-D surface plot of the potential")
surface = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=plt.cm.jet)
plt.savefig('plot_3dcontour.png')
plt.close()

#finding Current density
Jx,Jy = (0.5*(phi[1:-1,0:-2]-phi[1:-1,2:]),1/2*(phi[:-2,1:-1]-phi[2:,1:-1]))

#plotting current density
plot_current_density(Y,X,Jx,Jy)

#initialize temp
temp=300 * np.ones((Nx,Ny),dtype = float)

#the iterations
for k in range(Niter):
    tempold = temp.copy()
    temp = tempdef(temp,tempold,Jx,Jy)
    temp = temper(temp,ii)

#plotting 2d contour of final temp
plot_2dcontour_final(Y,X,temp)

