import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib import cm


Lengde_x = 2
Lengde_y = 2 
T = 0.05 

Punkt_x = 20
Punkt_y = 20
Tid = 100

h_x = Lengde_x/Punkt_x
h_y = Lengde_y/Punkt_y
k = T/Tid

x = np.linspace(0, Lengde_x, Punkt_x)
y = np.linspace(0,Lengde_y,Punkt_y)
t = np.linspace(0,T, Tid)

varme = np.zeros((Punkt_x,Punkt_y,Tid))


##Funksjon f her
f  = lambda x, y: np.sin((x+y)*np.pi)
alpha = 1.27 ##termisk diffusivitet

funksjonsverdier = np.array([[alpha*f(x_val, y_val)for y_val in y] for x_val in x])

varme[1:-1,1:-1,0]= funksjonsverdier[1:-1, 1:-1]




#metode 

def varme_next(varme):
    varme_next = np.zeros((Punkt_x, Punkt_y))
    for i in range(1, Punkt_x-1):
        for j in range(1,Punkt_y-1):
            varme_next[i,j] = k/(h_x**2)*(varme[i-1,j]-2*varme[i,j]+varme[i+1,j]) +k/(h_y**2)*(varme[i,j-1]-2*varme[i,j]+varme[i,j+1])+varme[i,j]
    return varme_next

for i in range(Tid-1):
    varme[:,:,i+1] = varme_next(varme[:,:,i])


##def varme_next(varme, k, h_x, h_y):
  ##  varme_next = np.zeros_like(varme)
    ##varme_next[1:-1, 1:-1] = k * ((varme[:-2, 1:-1] - 2*varme[1:-1, 1:-1] + varme[2:, 1:-1]) / h_x**2 +
      ##                            (varme[1:-1, :-2] - 2*varme[1:-1, 1:-1] + varme[1:-1, 2:]) / h_y**2)
    #return varme_next

#for i in range(Tid-1):
 #   varme[:, :, i+1] = varme_next(varme[:, :, i], k, h_x, h_y)

#Graf
#Frames per sekund
fps = 60

fig,ax =plt.subplots(subplot_kw={"projection": "3d"})

meshX,meshY = np.meshgrid(x,y)


def update(frame):
    ax.clear()
    surf = ax.plot_surface(meshX,meshY, varme[:,:,frame],cmap = cm.coolwarm)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$u$")
    ax.set_xlim((0,Lengde_x))
    ax.set_ylim((0,Lengde_y))
    ax.set_zlim((-1,1))
    return surf
animasjon = animation.FuncAnimation(fig, update, frames=np.linspace(0,Tid-1, fps).astype(int), interval = 1000/fps)

animasjon.save("Animation2d.gif")
plt.show()