import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_1 = 0
mu_2 = 2
sigma = 1
obs = 50

def sigmoid(x, mu, sigma):
    return (1 / (1 + np.exp( -(1/sigma)*(mu_2 - mu_1)*x + 0.5*( mu_2*mu_2*(1/sigma) - mu_1*mu_1*(1/sigma) )) ))

x_pos = np.random.normal(mu_1, sigma, obs)
x_neg = np.random.normal(mu_2, sigma, obs)
y = np.zeros((obs,1))

n_x_pos = np.linspace(norm.ppf(0.001,mu_1,sigma),norm.ppf(0.999, mu_1,sigma), 100)
n_x_neg = np.linspace(norm.ppf(0.001,mu_2,sigma),norm.ppf(0.999, mu_2,sigma), 100)

s_x= np.linspace(norm.ppf(0.001,mu_1,sigma),norm.ppf(0.999, mu_2,sigma), 200)

plt.figure(1, figsize=(4, 4))
plt.scatter(x_pos, y, marker="+", c="red", s=10)
plt.scatter(x_neg, y, marker="+", c="blue", s=10)
plt.plot(s_x, norm.pdf(s_x, mu_1, sigma),'r-', lw=3, alpha=0.6, label='P(x|y=0)')
plt.plot(s_x, norm.pdf(s_x, mu_2, sigma),'b-', lw=3, alpha=0.6, label='P(x|y=1)')
plt.plot(s_x, sigmoid(s_x,mu_2,sigma),'g-', lw=3, alpha=0.6, label='P(y=1|x)')
plt.plot((1, 1), (0, 0.6), 'k--')
plt.legend(loc=0)
plt.show()