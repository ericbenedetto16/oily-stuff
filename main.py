import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Graphing Stuff
resolutionX = 200
resolutionY = 200
tau = 0.5 # Viscosity
timeSteps = 4000
plotRealTime = True

# LBM Stuff
numberVelocities = 9
idxs = np.arange(numberVelocities)
eX = np.array([0,1,0,-1,0,1,-1,-1,1]) # Velocities X
eY = np.array([0,0,1,0,-1,1,1,-1,-1]) # Velocities Y
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # Weights

# Initial Conditions
F = np.zeros((resolutionX,resolutionY,9)) # Distribution Functions (x, y)

# Add Oil to Select Square in the Lattice
rho = np.zeros((resolutionX,resolutionY))
for i in range(100,120):
    for j in range(100,120):
        F[i,j,:] = 1
        rho[i,j] = 100

# Simulation Loop
for it in range(0,3000):
    # Streaming Process
    for i, cx, cy in zip(idxs, eX, eY):
        F[:,:,i] = np.roll(F[:,:,i], cx, axis=0) # Shift the X Axis in the Direction of the Velocity
        F[:,:,i] = np.roll(F[:,:,i], cy, axis=1) # Shift the Y Axis in the Direction of the Velocity

    # Calculate Densities and Velocities
    rho = np.sum(F,2)
    uX = np.sum(F*eX,2) / (rho + 0.01)
    uY = np.sum(F*eY,2) / (rho + 0.01)

    # Collision
    Feq = np.zeros(F.shape)
    for i, cX, cY, weight in zip(idxs, eX, eY, w):
        Feq[:,:,i] = rho * weight * (1 + 3*(cX*uX+cY*uY) + 9*(cX*uX+cY*uY)**2/2 - 3*(uX**2+uY**2)/2) # Apply Equilibrium Function

    F += -(1.0/tau) * (F - Feq)

    # Plot
    if(plotRealTime and (it % 10) == 0) or (it == timeSteps - 1):
        plt.cla()
        plt.imshow(rho, cmap='hot')
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.invert_yaxis()		
        plt.pause(1)

plt.savefig('latticeboltzmann.png',dpi=240)
plt.show()
