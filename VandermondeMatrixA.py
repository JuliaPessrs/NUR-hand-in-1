import numpy as np
import matplotlib.pyplot as plt
import sys
import os

data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1001)

def Matrix(x, N):
    '''Returns a Vandermonde matrix.'''
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i,j] = x[i]**j
    return M

def Sum(M, i, j, N):
    '''Returns the total sum.'''
    total = 0
    for k in range(0, N):
        total += M[i,k]*M[k,j]
    return total

def LU_matrix(Mat):
    '''Returns a LU matrix, containing both L and U.'''
    M = np.copy(Mat)
    nx, ny = M.shape
    for i in range(0,nx):
        for j in range(0,i):
            M[i,j] = (M[i,j] - Sum(M,i,j,j))/M[j,j]
        for j in range(i,ny):
            M[i,j] = M[i,j] - Sum(M,i,j,i)
    return M

def splitting(M):
    '''Returns to seperate L and U matrices.'''
    l = np.zeros_like(M)
    u = np.zeros_like(M)
    for i in range(len(l)):
        l[i,i] = 1
        for j in range(0,i):
            l[i,j] = M[i,j]
        for j in range(i,len(l)):
            u[i,j] = M[i,j]
    return l,u

def Forward_substitution(l, b):
    '''Applies forward substitution, where the input array b gets overwritten.'''
    for i in range(len(b)):
        for j in range(i):
            b[i] -= l[i,j]*b[j]
        b[i] /= l[i,i]
    return b

def Backward_substitution(u, b):
    '''Applies backward substitution, where the input array b gets overwritten.'''
    for i in range(len(b)-1,-1,-1): 
        for j in range(i+1,len(b)):
            b[i] -= u[i,j]*b[j]
        b[i] /= u[i,i]
    return b

def running(x,y):
    '''Returns the found values for c'''
    x_arr = np.copy(x)
    y_arr = np.copy(y)
    
    V = Matrix(x_arr,20)
    LU = LU_matrix(V)
    L, U = splitting(LU)
    z = Forward_substitution(L,y_arr)
    c = Backward_substitution(U,z)
    return V, LU, L, U, c

V_mat, LU_mat, L_mat, U_mat, c = running(x,y)
np.savetxt('Vandermode_a_output.txt', c)

yya = np.zeros_like(xx)
ya = np.zeros_like(x)
for j in range(len(c)):
    yya += c[j]*(xx**j)  
    ya += c[j]*(x**j)
    
fig=plt.figure()
gs=fig.add_gridspec(2,hspace=0,height_ratios=[2.0,1.0])
axs=gs.subplots(sharex=True,sharey=False)
axs[0].plot(x,y,marker='o',linewidth=0)
plt.xlim(-1,101)
axs[0].set_ylim(-400,400)
axs[0].set_ylabel('$y$')
axs[1].set_ylim(1e-16,1e1)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line,=axs[0].plot(xx,yya,color='orange')
line.set_label('Via LU decomposition')
axs[0].legend(frameon=False,loc="lower left")
axs[1].plot(x,abs(y-ya),color='orange')
plt.savefig('my_vandermonde_sol_2a.png',dpi=600)
plt.close()