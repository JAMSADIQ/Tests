import numpy as np
import matplotlib.pyplot as plt

#functions
def gauss(x, mu, sigma):
    return np.exp(-0.5 * (x - mu)**2 / sigma**2)

def fpop(x, mu, sigma, gp, gpp):
    return np.exp(gp * (x - mu) + 0.5 * gpp * (x-mu)**2)

def product_expsum(x, mu, sigma, gp, gpp):
    return np.exp((-0.5 * (x - mu)**2 / sigma**2) + (gp * (x - mu) + 0.5 * gpp * (x-mu)**2))

def productfunc(x, mu, sigma, gp, gpp):
    return gauss(x, mu, sigma) * fpop(x, mu, sigma, gp, gpp)

def CalculatedGaussForm(x, mu, sigma, gp, gpp):
    #compute constt, and mu + sigma of new gaussian
    constt = np.exp((gp * sigma)**2 / (2 * (1 - gpp * sigma**2)))
    new_mu = mu + (gp * sigma**2) / (1.0 - gpp * sigma**2)
    new_sigma_square = sigma**2 / (1.0 - gpp *sigma**2)
    expression =  constt * np.exp(-0.5 * (x - new_mu)**2 / new_sigma_square)
    #using gaussion func  
    # expression = const * gauss(x, new_mu, np.sqrt(new_sigma_square)) 
    return expression

mu = 1.0
sigma = 0.5
#let g = np.sin(mu)
gp = np.cos(mu)
gpp = - np.sin(mu)
xvals = np.linspace(-2, 4, 100)
#plots
gaussval = gauss(xvals, mu, sigma)
fpopval = fpop(xvals, mu, sigma, gp, gpp)
productval = productfunc(xvals, mu, sigma, gp, gpp)
myfuncval = CalculatedGaussForm(xvals,  mu, sigma, gp, gpp)
listf = [gaussval,fpopval,myfuncval, productval]
namef = ['gaussian','fpop', 'myexp','product']
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12,4))
for i, ax in enumerate(axs.flatten()):
    plt.sca(ax)
    plt.plot(xvals, listf[i], label=namef[i])
    plt.legend()
plt.show()

plt.figure()
plt.plot(xvals, listf[2], 'ko', label=namef[2])
plt.plot(xvals, listf[3], 'r--', label=namef[3])
plt.legend()
plt.show()
