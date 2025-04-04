import numpy as np
from scipy.integrate import quad

class HestonModel:
    def __init__(self, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        """
        Heston model parameters:
        S0: Initial stock price
        K: Strike price
        v0: Initial variance
        kappa: Rate of mean reversion for variance
        theta: Long-term variance
        sigma: Volatility of variance
        rho: Correlation between stock and variance
        lambd: Risk premium
        tau: Time to maturity
        r: Risk-free rate
        """
        self.S0 = S0
        self.K = K
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lambd = lambd
        self.tau = tau
        self.r = r
    
    def heston_charfunc(self, phi):
        """Computes the characteristic function of the Heston model."""
        a = self.kappa * self.theta
        b = self.kappa + self.lambd
        rspi = self.rho * self.sigma * phi * 1j
        d = np.sqrt((rspi - b) ** 2 + (phi * 1j + phi ** 2) * self.sigma ** 2)
        g = (b - rspi + d) / (b - rspi - d)
        
        exp1 = np.exp(self.r * phi * 1j * self.tau)
        term2 = self.S0 ** (phi * 1j) * ((1 - g * np.exp(d * self.tau)) / (1 - g)) ** (-2 * a / self.sigma ** 2)
        exp2 = np.exp(a * self.tau * (b - rspi + d) / self.sigma ** 2 + \
                      self.v0 * (b - rspi + d) * ((1 - np.exp(d * self.tau)) / (1 - g * np.exp(d * self.tau))) / self.sigma ** 2)
        
        return exp1 * term2 * exp2
    
    def integrand(self, phi):
        """Defines the integrand function for the Heston model price calculation."""
        numerator = np.exp(self.r * self.tau) * self.heston_charfunc(phi - 1j) - self.K * self.heston_charfunc(phi)
        denominator = 1j * phi * self.K ** (1j * phi)
        return numerator / denominator
    
    def heston_price(self, option_type="call"):
        """Computes the price of a European call or put option using the Heston model."""
        real_integral, err = quad(lambda phi: np.real(self.integrand(phi)), 0, 100)
        #real_integral, err = np.real(quad(self.integrand, 0, 100))
        call_price = (self.S0 - self.K * np.exp(-self.r * self.tau)) / 2 + real_integral / np.pi
        if option_type.lower() == "call":
            return call_price
        elif option_type.lower() == "put":
            return call_price + self.K * np.exp(-self.r * self.tau) - self.S0
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")


