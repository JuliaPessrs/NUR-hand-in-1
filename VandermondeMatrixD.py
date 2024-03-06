import timeit

setup = '''
import numpy as np
import sys
import os

data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1001)
'''

a_code = '''
def Matrix(x, N):
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i,j] = x[i]**j
    return M
    
def Sum(M, i, j, N):
    total = 0
    for k in range(0, N):
        total += M[i,k]*M[k,j]
    return total

def LU_matrix(Mat):
    M = np.copy(Mat)
    nx, ny = M.shape
    for i in range(0,nx):
        for j in range(0,i):
            M[i,j] = (M[i,j] - Sum(M,i,j,j))/M[j,j]
        for j in range(i,ny):
            M[i,j] = M[i,j] - Sum(M,i,j,i)
    return M

def splitting(M):
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
    for i in range(len(b)):
        for j in range(i):
            b[i] -= l[i,j]*b[j]
        b[i] /= l[i,i]
    return b

def Backward_substitution(u, b):
    for i in range(len(b)-1,-1,-1): 
        for j in range(i+1,len(b)):
            b[i] -= u[i,j]*b[j]
        b[i] /= u[i,i]
    return b

def running(x,y):
    x_arr = np.copy(x)
    y_arr = np.copy(y)
    
    V = Matrix(x_arr,20)
    LU = LU_matrix(V)
    L, U = splitting(LU)
    z = Forward_substitution(L,y_arr)
    c = Backward_substitution(U,z)
    return V, LU, L, U, c

V_mat, LU_mat, L_mat, U_mat, c = running(x,y)
yya = np.zeros_like(xx)
ya = np.zeros_like(x)
for j in range(len(c)):
    yya += c[j]*(xx**j)  
    ya += c[j]*(x**j)
'''
a_time = timeit.timeit(a_code, setup, number=200)

b_code = '''
def sample_points(x, data, M):
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
    yb +=[Nevilles_algorithm(x[i], x, y, 20)]'''

b_time = timeit.timeit(b_code, setup, number=200)

c_code = '''
def Matrix(x, N):
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i,j] = x[i]**j
    return M
    
def Sum(M, i, j, N):
    total = 0
    for k in range(0, N):
        total += M[i,k]*M[k,j]
    return total

def LU_matrix(Mat):
    M = np.copy(Mat)
    nx, ny = M.shape
    for i in range(0,nx):
        for j in range(0,i):
            M[i,j] = (M[i,j] - Sum(M,i,j,j))*(1/M[j,j])
        for j in range(i,ny):
            M[i,j] = M[i,j] - Sum(M,i,j,i)
    return M

def splitting(M):
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
    for i in range(len(b)):
        for j in range(i):
            b[i] -= l[i,j]*b[j]
        b[i] *= (1/l[i,i])
    return b

def Backward_substitution(u, b):
    for i in range(len(b)-1,-1,-1): 
        for j in range(i+1,len(b)):
            b[i] -= u[i,j]*b[j]
        b[i] *= (1/u[i,i])
    return b
    
def product(M, b):
    nx, ny = M.shape
    dot = np.zeros_like(b)
    for i in range(ny):
        for j in range(nx):
            dot[i] += M[i,j]*b[j]
    return dot

def interations(num, M, l, u, c, y):
    b = np.copy(c)
    for i in range(num):
        dy = product(M,b) - y
        z = Forward_substitution(l,dy)
        v = Backward_substitution(u, z)
        b -= v
    return b
    
def running(x,y):
    x_arr = np.copy(x)
    y_arr = np.copy(y)
    
    V = Matrix(x_arr,20)
    LU = LU_matrix(V)
    L, U = splitting(LU)
    z = Forward_substitution(L,y_arr)
    c = Backward_substitution(U,z)
    return V, LU, L, U, c
    
V_mat, LU_mat, L_mat, U_mat, c = running(x,y)
c10 = interations(10, V_mat, L_mat, U_mat, c, y)
yyc10 = np.zeros_like(xx)
yc10 = np.zeros_like(x)
for j in range(len(c10)):
    yyc10 += c10[j]*(xx**j)  
    yc10 += c10[j]*(x**j)
'''
c_time = timeit.timeit(c_code, setup, number=200)

np.savetxt('Vandermode_d_output.txt', np.array([a_time, b_time, c_time]))
