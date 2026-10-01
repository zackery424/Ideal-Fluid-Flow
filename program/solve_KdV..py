import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation


#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----

def RK4Step(yprime,t,y,dt):
    k1 = yprime(t       , y          )
    k2 = yprime(t+0.5*dt, y+0.5*k1*dt)
    k3 = yprime(t+0.5*dt, y+0.5*k2*dt)
    k4 = yprime(t+    dt, y+    k3*dt)
    return y + (k1+2*k2+2*k3+k4)*dt/6. 

def RK4(yprime,times,y0):
    y = np.empty(times.shape+y0.shape,dtype=y0.dtype)
    y[0,:] = y0                         # enter initial conditions in y
    steps = 4
    for i in range(times.size-1):
        dt = (times[i+1]-times[i])/steps
        y[i+1,:] = y[i,:]
        for k in range(steps):
            y[i+1,:] = RK4Step(yprime, times[i]+k*dt, y[i+1,:], dt)
            y[i+1,:] = y[i+1,:] * np.exp(delta_t * 1j * kx**3)
    return y

#====================================================================
#----- Parameters for PDE -----
L       = 100
n       = 512
delta_t = 0.1
tmax    = 20
c1      = 1.5  # amplitude of 1st wave
c2      = 0.5  # amplitude of 2nd wave

#----- Constructing the grid and kernel functions
x2      = np.linspace(-L/2,L/2, n+1)
x       = x2[:n]                         # periodic B.C. #0 = #n

kx1     = np.linspace(0,n//2-1,n//2)
kx2     = np.linspace(1,n/2,  n//2)
kx2     = -1*kx2[::-1]
kx      = (2.* np.pi/L)*np.concatenate((kx1,kx2)); kx[0] = 1e-6

#----- Initial Condition -----
z1      = np.sqrt(c1)/2. * (x-0.1*L)
z2      = np.sqrt(c2)/2. * (x-0.4*L)
soliton = 6*(0.5 * c1 / (np.cosh(z1))**2 + 0.5 * c2/ (np.cosh(z2))**2)
uhat0   = np.fft.fft(soliton)
vhat0   = uhat0

#----- Define ODE -----
def wprime(t,vhat):  
    g = -0.5 * 1j* kx * np.exp(-1j * kx**3 * t)
    return g * np.fft.fft(np.real(np.fft.ifft(np.exp(1j*kx**3*t)*vhat)**2))

#====================================================================
#----- Compute the numerical solution -----
TimeStart = 0.
TimeEnd   = tmax+delta_t
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation


#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----

def RK4Step(yprime,t,y,dt):
    k1 = yprime(t       , y          )
    k2 = yprime(t+0.5*dt, y+0.5*k1*dt)
    k3 = yprime(t+0.5*dt, y+0.5*k2*dt)
    k4 = yprime(t+    dt, y+    k3*dt)
    return y + (k1+2*k2+2*k3+k4)*dt/6. 

def RK4(yprime,times,y0):
    y = np.empty(times.shape+y0.shape,dtype=y0.dtype)
    y[0,:] = y0                         # enter initial conditions in y
    steps = 4
    for i in range(times.size-1):
        dt = (times[i+1]-times[i])/steps
        y[i+1,:] = y[i,:]
        for k in range(steps):
            y[i+1,:] = RK4Step(yprime, times[i]+k*dt, y[i+1,:], dt)
            y[i+1,:] = y[i+1,:] * np.exp(delta_t * 1j * kx**3)
    return y

#====================================================================
#----- Parameters for PDE -----
L       = 100
n       = 512
delta_t = 0.1
tmax    = 20
c1      = 1.5  # amplitude of 1st wave
c2      = 0.5  # amplitude of 2nd wave

#----- Constructing the grid and kernel functions
x2      = np.linspace(-L/2,L/2, n+1)
x       = x2[:n]                         # periodic B.C. #0 = #n

kx1     = np.linspace(0,n//2-1,n//2)
kx2     = np.linspace(1,n//2,  n//2)
kx2     = -1*kx2[::-1]
kx      = (2.* np.pi/L)*np.concatenate((kx1,kx2)); kx[0] = 1e-6

#----- Initial Condition -----
z1      = np.sqrt(c1)/2. * (x-0.1*L)
z2      = np.sqrt(c2)/2. * (x-0.4*L)
soliton = 6*(0.5 * c1 / (np.cosh(z1))**2 + 0.5 * c2/ (np.cosh(z2))**2)
uhat0   = np.fft.fft(soliton)
vhat0   = uhat0

#----- Define ODE -----
def wprime(t,vhat):  
    g = -0.5 * 1j* kx * np.exp(-1j * kx**3 * t)
    return g * np.fft.fft(np.real(np.fft.ifft(np.exp(1j*kx**3*t)*vhat)**2))

#====================================================================
#----- Compute the numerical solution -----
TimeStart = 0.
TimeEnd   = tmax+delta_t
TimeSpan  = np.arange(TimeStart, TimeEnd, delta_t)
w_sol     = RK4(wprime,TimeSpan, vhat0)

# #----- Animate the numerical solution -----
# fig = plt.figure()
# ims = []
# for i in TimeSpan:
#     w = np.real(np.fft.ifft(w_sol[i,:]))
#     im = plt.plot(x,w)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False)

# plt.show()
#----- Animate the numerical solution -----
fig = plt.figure()
ax = plt.axes(ylim=(-5,5))
line, = ax.plot(x, soliton, '-')

NT = TimeSpan.size;

def animate(i):
    i = i % (NT + 10)
    if i<NT:
        w = np.real(np.fft.ifft(w_sol[i,:]))
        line.set_ydata(w)
    return line,


anim = animation.FuncAnimation(fig, animate, interval=50, blit=True)
# anim.save('KdV.mp4', dpi = 300)

plt.show()