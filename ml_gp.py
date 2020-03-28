import numpy as np 
import reg as reg
import pylab as pl

np.random.seed(123190012)

nd = 1
nobs = 200

w = 1 + np.random.rand(nd)
f_sig2 = .001

Xobs = np.random.rand(nd, nobs)
yobs = np.array([np.dot(w,Xobs[:, i]) + f_sig2**.5*np.random.normal() for i in range(nobs)])

w_prior_cov = np.eye(nd)
linear_reg = reg.linear_regression(nd, w_prior_cov, f_sig2)

w_post_mu, w_post_cov = linear_reg.update(Xobs, yobs)

print("(w, f_sig2) = (%g, %g)" %(w, f_sig2))
print("nobs = %d" %nobs)
print("w_prior_cov = %g" %w_prior_cov)
print("(w_post_mu, w_post_cov) = (%g, %g)" %(w_post_mu, w_post_cov))

if (nd == 1):
  ax = pl.subplot()
  ax.plot(Xobs.T, yobs, ".")
  _x = np.linspace(0, 1, 1000)
  ax.plot(_x, _x*w_post_mu)
  pl.show()