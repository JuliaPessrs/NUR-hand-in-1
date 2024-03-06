def product(M, b):
    '''Returns a 1D array after multiplying a matrix with a vector.'''
    nx, ny = M.shape
    dot = np.zeros_like(b)
    for i in range(ny):
        for j in range(nx):
            dot[i] += M[i,j]*b[j]
    return dot

def interations(num, M, l, u, c, y):
    '''Returns b after applying num interations.'''
    b = np.copy(c)
    for i in range(num):
        dy = product(M,b) - y
        z = Forward_substitution(l,dy)
        v = Backward_substitution(u, z)
        b -= v
    return b

c1 = interations(1, V_mat, L_mat, U_mat, c, y)
c10 = interations(10, V_mat, L_mat, U_mat, c, y)

yyc1 = np.zeros_like(xx)
yc1 = np.zeros_like(x)
yyc10 = np.zeros_like(xx)
yc10 = np.zeros_like(x)
for j in range(len(c1)):
    yyc1 += c1[j]*(xx**j)  
    yc1 += c1[j]*(x**j)
    yyc10 += c10[j]*(xx**j)  
    yc10 += c10[j]*(x**j)
    
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
#plt.savefig('my_vandermonde_sol_2a.png',dpi=600)
line,=axs[0].plot(xx,yyb,linestyle='dashed',color='green')
line.set_label('Via Neville\'s algorithm')
axs[0].legend(frameon=False,loc="lower left")
axs[1].plot(x,abs(y-yb),linestyle='dashed',color='green')
#plt.savefig('my_vandermonde_sol_2b.png',dpi=600)
line,=axs[0].plot(xx,yyc1,linestyle='dotted',color='red')
line.set_label('LU with 1 iteration')
axs[1].plot(x,abs(y-yc1),linestyle='dotted',color='red')
line,=axs[0].plot(xx,yyc10,linestyle='dashdot',color='purple')
line.set_label('LU with 10 iterations')
axs[1].plot(x,abs(y-yc10),linestyle='dashdot',color='purple')
axs[0].legend(frameon=False,loc="lower left")
plt.savefig('my_vandermonde_sol_2c.png',dpi=600)
plt.close()