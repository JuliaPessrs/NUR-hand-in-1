import numpy as np

def Poisson(l, k):
    '''Returns the outcome of the Poission distribution.
    Depending on the number of k, the outcome may be calculated in log space to prevent overflow.'''
    P = np.float32()
    if k < 20:
        P = ((l**k)*(np.exp(-l)))/factorial(k)
    else:
        logP = k*np.log(l) - l - logFactorial(k)
        P = np.exp(logP)
    return P

def factorial(k):
    '''Returns the factorial of the input.'''
    factors = np.arange(1, k+1, dtype = np.float32)
    fact = 1
    for number in factors:
        fact *= number
    return fact

def logFactorial(k):
    '''Returns the log of the factorial of the input.
    For high values of k the factorial will be determined in log space to prevent overflow.'''
    factors = np.arange(1, k+1, dtype = np.float32)
    logFact = 0
    for number in factors:
        logFact += np.log(number)
    return logFact

results = np.zeros((5,1), dtype = np.float32)

results[0] = Poisson(1,0)
results[1] = Poisson(5,10)
results[2] = Poisson(3,21)
results[3] = Poisson(2.6,40)
results[4] = Poisson(101,200)

np.savetxt('Poisson_output.txt', results)








