import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch

from outils import Spec_Mix as SM
from outils import outersum


class gp_class:

    def init_hypers(self, case='canonical'):
        self.x = None
        self.y = None
        if case == 'random':
            self.sigma = np.random.rand()*2
            self.gamma = np.random.rand()
            self.mu = np.random.rand()*0.1
            self.sigma_n = np.random.rand()*0.1
        elif case == 'canonical':
            self.sigma = 10
            self.gamma = 1/2
            self.mu = 0.1
            self.sigma_n = 0.1
            
    def show_hypers(self):
        print(f'gamma: {self.gamma}, i.e., lengthscale = {np.sqrt(1/(2*self.gamma))}')
        print(f'sigma: {self.sigma}')
        print(f'sigma_n: {self.sigma_n}')
        print(f'mu: {self.mu}')
        
        
    def sample(self, how_many=1):
        samples = np.random.multivariate_normal(self.mean, self.cov, size=how_many)
        self.samples = samples.T
        return self.samples

        
    def plot_samples(self, linestyle='-', v_axis_lims=None):
        if v_axis_lims == None:
            v_axis_lims = np.max(np.abs(self.samples))
        plt.figure(figsize=(9, 4))
        error_bars = 2 * self.sigma
        plt.fill_between(self.time, - error_bars, error_bars,
                         color='blue', alpha=0.1, label='95% barres d\'erreur')
        plt.plot(self.time, np.zeros_like(
            self.time), alpha=0.7, label='moyenne')
        if self.samples.shape[1] == 1:
            plt.plot(self.time, self.samples, linestyle, c='r', alpha=1)
        else:
            plt.plot(self.time, self.samples, linestyle, alpha=1)
        plt.title('observations du GP')
        plt.xlabel('temps')
        plt.legend(loc=1)
        plt.xlim([min(self.time), max(self.time)])
        plt.ylim([-v_axis_lims, v_axis_lims])
        plt.tight_layout()
        
        
    def load(self, x, y):
        self.Nx = len(x)
        self.x = x
        self.y = y
        
        
        
    def plot_data(self):
            
        plt.figure(figsize=(9, 3))
        plt.plot(self.x, self.y, '.r', markersize=8, label='données')
        plt.xlabel('temps')
        plt.legend(loc=1)
        plt.xlim([min(self.x), max(self.x)])
        plt.tight_layout()
        
        
    def plot_posterior(self, n_samples=0, v_axis_lims=None):

        plt.figure(figsize=(9, 3))
        plt.plot(self.time, self.mean, 'b', label='posterieur')

        plt.plot(self.x, self.y, '.r', markersize=8, label='données')
        error_bars = 2 * np.sqrt((np.diag(self.cov)))
        plt.fill_between(self.time, self.mean - error_bars, self.mean +
                         error_bars, color='blue', alpha=0.1, label='95% barres d\'erreur')
        if n_samples > 0:
            self.compute_posterior(where=self.time)
            self.sample(how_many=n_samples)
            plt.plot(self.time, self.samples, alpha=0.7)
        plt.title('Posterieur')
        plt.xlabel('temps')
        plt.legend(loc=1, ncol=3)
        plt.xlim([min(self.x), max(self.x)])
        plt.ylim([-35, 35])
        plt.tight_layout()

    def compute_posterior(self, dimension=None, where=None):
      if dimension is None:
          self.N = 100
          self.time = np.linspace(1, 100, 100)
      elif np.size(dimension) == 1:
          self.N = dimension
          self.time = np.linspace(1, 100, dimension)
      if where is not None:
          self.time = where
          self.N = len(where)

      if self.x is None:  # nous n'avons aucune observations
          self.mean = np.zeros(self.N)
          self.cov = SM(self.time, self.time, self.gamma, self.mu, self.sigma) + ((self.sigma_n)**2)*np.eye(self.N)
      else:  # nous avons accès à des observations
          cov_obs = SM(self.x, self.x, self.gamma, self.mu, self.sigma) + (1e-5 + (self.sigma_n)**2)*np.eye(self.Nx)
          cov_star = SM(self.time, self.x, self.gamma, self.mu, self.sigma) 
          cov_star_star = SM(self.time, self.time, self.gamma, self.mu, self.sigma) + (1e-5)*np.eye(self.N)
          self.mean = np.squeeze(cov_star @ np.linalg.solve(cov_obs, self.y))
          self.cov = cov_star_star - cov_star @ np.linalg.inv(cov_obs) @ cov_star.T
              
    def nll(self):
        Y = self.y
        K = SM(self.x, self.x, self.gamma, self.mu, self.sigma) + ((self.sigma_n)**2)*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*logdet + 0.5*(Y.T)@ np.linalg.inv(K) @ Y + 0.5*self.Nx*np.log(2*np.pi)

    def nlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        mu = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        K = SM(self.x, self.x, gamma, mu, sigma) + ((sigma_n)**2)*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*logdet + 0.5*(Y.T)@ np.linalg.inv(K) @ Y + 0.5*self.Nx*np.log(2*np.pi)

    def nlogp(self, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        mu = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        K = SM(self.x, self.x, gamma, mu, sigma) + ((sigma_n)**2)*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*logdet + 0.5*(Y.T)@ np.linalg.inv(K) @ Y + 0.5*self.Nx*np.log(2*np.pi)

    def nlogp_v2(self,hypers) : 
    # Conversion en tenseurs si nécessaire
      if not isinstance(hypers[0], torch.Tensor):
          sigma = torch.exp(torch.tensor(hypers[0], dtype=torch.float64))
          gamma = torch.exp(torch.tensor(hypers[1], dtype=torch.float64))
          mu = torch.exp(torch.tensor(hypers[2], dtype=torch.float64))
          sigma_n = torch.exp(torch.tensor(hypers[3], dtype=torch.float64))
      else:
          sigma = torch.exp(hypers[0])
          gamma = torch.exp(hypers[1])
          mu = torch.exp(hypers[2])
          sigma_n = torch.exp(hypers[3])
      
      # Conversion des données en tenseurs PyTorch
      Y = torch.tensor(self.y, dtype=torch.float64) if not isinstance(self.y, torch.Tensor) else self.y
      x_tensor = torch.tensor(self.x, dtype=torch.float64) if not isinstance(self.x, torch.Tensor) else self.x
      
      # Calcul de la matrice de covariance
      K = SM(x_tensor, x_tensor, gamma, mu, sigma) + (sigma_n**2) * torch.eye(self.Nx, dtype=torch.float64)
      
      # Calcul du log-déterminant (PyTorch)
      logdet = torch.logdet(K)
      
      # Calcul de la vraisemblance négative
      # Utilisation de torch.linalg.solve au lieu de np.linalg.inv pour la stabilité numérique
      K_inv_Y = torch.linalg.solve(K, Y.reshape(-1, 1))
      quadratic_term = Y.reshape(1, -1) @ K_inv_Y
      
      loss = 0.5 * logdet + 0.5 * quadratic_term.squeeze() + 0.5 * self.Nx * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
      
      return loss
      
    def dnlogp(self, hypers):
      sigma = np.exp(hypers[0])
      gamma = np.exp(hypers[1])
      mu = np.exp(hypers[2])
      sigma_n = np.exp(hypers[3])
      K = SM(self.x, self.x, gamma, mu, sigma)/sigma
      norm = outersum(self.x,-self.x)


      ## Dérivées par rapport aux paramètres
      dKdsigma = 2*sigma*np.exp(-gamma*(np.subtract.outer(self.x, self.x)**2))*np.cos(2*np.pi*mu*np.abs(np.subtract.outer(self.x, self.x)))
      dKdgamma = -sigma**2 * (np.subtract.outer(self.x, self.x)**2) * np.exp(-gamma*(np.subtract.outer(self.x, self.x)**2)) * np.cos(2*np.pi*mu*np.abs(np.subtract.outer(self.x, self.x)))
      dKdmu = -2*np.pi*sigma**2 * np.abs(np.subtract.outer(self.x, self.x)) * np.exp(-gamma*(np.subtract.outer(self.x, self.x)**2)) * np.sin(2*np.pi*mu*np.abs(np.subtract.outer(self.x, self.x)))
      dKdsigma_n = 2*sigma_n * np.eye(len(self.x))

      K  = SM(self.x, self.x, gamma, mu, sigma) + (sigma_n**2)*np.eye(len(self.x))
      dlogp_dsigma = 0.5 * np.trace(np.linalg.inv(K) @ dKdsigma) - 0.5 * self.y.T @ np.linalg.inv(K) @ dKdsigma @ np.linalg.inv(K) @ self.y
      dlogp_dgamma = 0.5 * np.trace(np.linalg.inv(K) @ dKdgamma) - 0.5 * self.y.T @ np.linalg.inv(K) @ dKdgamma @ np.linalg.inv(K) @ self.y
      dlogp_dmu = 0.5 * np.trace(np.linalg.inv(K) @ dKdmu) - 0.5 * self.y.T @ np.linalg.inv(K) @ dKdmu @ np.linalg.inv(K) @ self.y
      dlogp_dsigma_n = 0.5 * np.trace(np.linalg.inv(K) @ dKdsigma_n) - 0.5 * self.y.T @ np.linalg.inv(K) @ dKdsigma_n @ np.linalg.inv(K) @ self.y
      return np.array([dlogp_dsigma, dlogp_dgamma, dlogp_dmu, dlogp_dsigma_n])

    def train(self, flag='quiet'):
      hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.mu), np.log(self.sigma_n)])
      res = minimize(self.nlogp, hypers0)
      self.sigma = np.exp(res.x[0])
      self.gamma = np.exp(res.x[1])
      self.mu = np.exp(res.x[2])
      self.sigma_n = np.exp(res.x[3])
      self.theta = np.array([self.mu, self.gamma, self.sigma_n])
      if flag != 'quiet':
          print('Hyperparameters are:')
          print(f'sigma ={self.sigma}')
          print(f'gamma ={self.gamma}')
          print(f'mu ={self.mu}')
          print(f'sigma_n ={self.sigma_n}')


