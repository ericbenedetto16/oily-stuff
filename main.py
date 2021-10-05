import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# Logs
# streaminglog = open("./logs/streaming.log", mode="a")
# macrolog = open("./logs/macro.log", mode="a")
# feqlog = open("./logs/feq.log", mode="a")
# collisionlog = open("./logs/collision.log", mode="a")

# Graphing Stuff
resolutionX = 10
resolutionY = 10
timeSteps = 5
plotRealTime = True
plotInterval = 1

# LBM Stuff
numberVelocities = 9
tau = 5.0 # Viscosity
idxs = np.arange(numberVelocities)
eX = np.array([0,1,0,-1,0,1,-1,-1,1]) # Velocities X
eY = np.array([0,0,1,0,-1,1,1,-1,-1]) # Velocities Y
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) # Weights

# Initial Conditions
F = np.zeros((resolutionX,resolutionY,9), dtype=float)
Feq = np.zeros(F.shape,  dtype=float)
rho = np.zeros((resolutionX, resolutionY),  dtype=float)
uX = np.zeros(rho.shape,  dtype=float)
uY = np.zeros(rho.shape,  dtype=float)

# Add Oil to Select Square in the Lattice
for i in range(4,6):
    for j in range(4,6):
        rho[i,j] = 1

for i, cX, cY, weight in zip(idxs, eX, eY, w):
    Feq[:,:,i] = rho * weight * (1 + 3*(cX*uX+cY*uY) + 9*(cX*uX+cY*uY)**2/2 - 3*(uX**2+uY**2)/2) # Apply Equilibrium Function

F = Feq.copy()

# Simulation Loop
for it in range(0, timeSteps + 1):
    # Streaming Process
    for i, cx, cy in zip(idxs, eX, eY):
        F[:,:,i] = np.roll(F[:,:,i], cx, axis=0) # Shift the X Axis in the Direction of the Velocity
        F[:,:,i] = np.roll(F[:,:,i], cy, axis=1) # Shift the Y Axis in the Direction of the Velocity

    # streaminglog.write(f"### Step {it} ###\n{F}\n\n")
    
    # Calculate Macroscopic Densities and Velocities
    rho = np.sum(F, 2)

    uX = np.divide(np.sum(F*eX,2), rho, out=np.zeros_like(rho), where=rho!=0)
    uY = np.divide(np.sum(F*eY,2), rho, out=np.zeros_like(rho), where=rho!=0)

    # macrolog.write(f"### Step {it} ###\n*** RHO ***\n{rho}\n*** u_x ***\n{uX}\n***u_y***\n{uY}\n\n")

    # Equilibrium
    for i, cX, cY, weight in zip(idxs, eX, eY, w):
        Feq[:,:,i] = rho * weight * (1 + 3*(cX*uX+cY*uY) + 9*(cX*uX+cY*uY)**2/2 - 3*(uX**2+uY**2)/2) # Apply Equilibrium Function

    # feqlog.write(f"### Step {it} ###\n{Feq}\n\n")

    # Collision   
    F -= (F - Feq) / tau

    # collisionlog.write(f"### Step {it} ###\n{F}\n\n")

    # Plot
    if(plotRealTime and (it % plotInterval) == 0) or (it == timeSteps - 1):
        # plt.cla()
        # plt.imshow(rho, cmap='hot')
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.invert_yaxis()		
        # plt.pause(0.001)

        print("t= ",it)
        
        plt.clf()
        # plot macro density result
        x = np.arange(0, resolutionX-1, 1)
        y = np.arange(0, resolutionY-1, 1)
        X, Y = np.meshgrid(x, y)
        Z = rho[X,Y] 
       
        
        cmap = colors.ListedColormap(['skyblue', 'saddlebrown'])
        # im = plt.imshow(Z, cmap=plt.cm.afmhot, extent=(0, resolutionX-1, 0, resolutionY-1),vmin=0, vmax=.0016)  
        im = plt.imshow(Z, cmap=cmap, extent=(0, resolutionX-1, 0, resolutionY-1),vmin=0, vmax=.0016)  

        plt.colorbar(im)  
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        
        plt.title('rho at step: ' + str(it))
        plt.pause(0.01)

# plt.savefig('latticeboltzmann.png',dpi=240)
plt.show()
