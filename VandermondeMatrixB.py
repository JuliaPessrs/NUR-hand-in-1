def sample_points(x, data, M):
    '''Returns the first value of the M data points surrounding x.'''
    n = len(data)
    data_copy = np.copy(data)
    
    for i in range(n): #checks if the input x is equal to a point in the data
        if x == data[i]:
            j_low = int(i) - int(M/2)
            if j_low < 0:
                j_low = 0
            return j_low
    
    while (n != 2): #finds the data points left and right of x
        split = n*0.5
        split = round(split)
        if (x < data[split]):
            if (x < data[split]) and (x > data[split-1]):
                data = data[split-1:split+1]
            else:
                data = data[:split]    
        else:
            data = data[split:]
        n = len(data)
    
    for i in range(len(data_copy)): #determines j_low
        if (data_copy[i] == data[0]):
            j_low = int(i) - int(M/2) + 1
            if j_low < 0:
                j_low = 0
    return j_low

def Nevilles_algorithm(x, x_data, y_data, M):
    '''Retuns interpolated data point after applyin Neville's algorithm''' 
    n = len(x_data)
    P = np.zeros(1)
    x_copy = np.copy(x_data)
    y_copy = np.copy(y_data)
    
    if (x <= x_data[0]): #checks if x lies inside the data points
        P[0] = y_data[0]
        
    if (x >= x_data[n-1]):
        P[0] = y_data[n-1]
        
    if (x > x_data[0]) and (x < x_data[n-1]):
        j_low = sample_points(x, x_data, M)
        x_data = x_copy[j_low:(j_low + M)]
        y_data = y_copy[j_low:(j_low + M)]
        if len(y_data) < M:
            z = M - len(y_data)
            j_low = j_low - z 
            x_data = x_copy[j_low:(j_low + M)]
            y_data = y_copy[j_low:(j_low + M)]
            
        P = np.copy(y_data)
        for k in range(1,len(P)):
            for i in range(0, len(P)-k):
                P[i] = ((x - x_data[i+k])*P[i] + (x_data[i] - x)*P[i+1])/(x_data[i]-x_data[i+k])    
    return P[0]

yyb = []
yb = []
for i in range(len(xx)):
    yyb +=[Nevilles_algorithm(xx[i], x, y, 20)]
for i in range(len(x)):
    yb +=[Nevilles_algorithm(x[i], x, y, 20)]
    
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
plt.savefig('my_vandermonde_sol_2b.png',dpi=600)
plt.close()