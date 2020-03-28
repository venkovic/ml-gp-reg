import numpy as np

class linear_regression():
  def __init__(self, nd, w_cov, f_sig2):
  	self.nd = nd
  	if np.isscalar(w_cov):
  	  w_cov = w_cov*np.eye(nd)
  	self.w_cov = w_cov
  	self.w_pre = np.linalg.inv(w_cov)
  	self.f_sig2 = f_sig2

  def update(self, Xobs, yobs):
  	self.Xobs = Xobs
  	self.yobs = yobs
  	_A = self.f_sig2**-1*Xobs.dot(Xobs.T) + self.w_pre
  	_invA = np.linalg.inv(_A)
  	self.w_mu = self.f_sig2**-1*_invA.dot(Xobs.dot(yobs))
  	self.w_cov = _invA
  	return self.w_mu, self.w_cov

class polynomial_regression():
  def __init__(self, nd, m, w_cov, f_sig2):
    self.nd = nd
    if (nd > 1):
      raise NameError("Only works for nd = 1.")
    self.m = m
    if np.isscalar(w_cov):
      w_cov = w_cov*np.eye(nd)
    self.w_cov = w_cov
    self.w_pre = np.linalg.inv(w_cov)
    self.f_sig2 = f_sig2

  def phi(self, x):
    _phi = x**np.linspace(0, self.m, self.m+1)
    return _phi

  def update(self, Xobs, yobs):
    self.Xobs = Xobs
    self.yobs = yobs
    nobs = Xobs.shape[1]
    self.Phi = np.zeros((self.nd*(self.m+1), nobs))
    for obs in range(nobs):
      self.Phi[:, obs] = self.phi(Xobs[:, obs])
    self.A = self.f_sig2**-1*self.Phi.dot(self.Phi.T) + self.w_pre
    self.invA = np.linalg.inv(self.A)
    return 

  def mu(self, x):
    phi_x = self.phi(x)
    return self.f_sig2**-1*phi_x.T.dot(self.invA.dot(self.Phi.dot(self.yobs)))


class gp_regression():
  def __init__(self, nd, f_sig2, ell, nu):
    self.nd = nd
    self.f_sig2 = f_sig2
    self.ell = ell
    self.nu = nu

  def cov(self, X, Y):
    nx, ny = X.shape[1], Y.shape[1]
    cov = np.zeros((nx, ny))
    for i in range(nx):
      for j in range(ny):
        dx = np.abs(X[:, i] - Y[:, j])/self.ell
        if (self.nd == 1):
          cov[i, j] = self.f_sig2*np.exp(-.5*dx**self.nu)
    return cov

  def update(self, X, f):
    self.X = X
    self.f = f
    nobs = X.shape[1]
    Kobs = self.cov(X, X)
    self.invKobs = np.linalg.inv(Kobs)

  def prediction(self, Xnew):
    Knew_obs = self.cov(Xnew, self.X)
    Knew_obs_invK = Knew_obs.dot(self.invKobs)
    mu = Knew_obs_invK.dot(self.f)
    cov = self.cov(Xnew, Xnew)
    cov -= Knew_obs_invK.dot(Knew_obs.T)
    return mu, cov
