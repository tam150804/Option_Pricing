# Third party imports
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

# Local package imports
from .base import PricingModel

class MonteCarloSimulation(PricingModel):
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    Simulates underlying asset prices at the expiry date using a random stochastic process (Brownian motion).
    Payoffs are calculated from simulated prices, averaged, and discounted to present value, representing the option price.
    """

    def __init__(self, spot_price, strike_price, maturity_days, risk_free_rate, volatility, num_simulations):
        """
        Initializes variables used in the Monte Carlo simulation for option pricing.

        spot_price: current stock or other underlying spot price
        strike_price: strike price for the option contract
        maturity_days: number of days until option contract maturity
        risk_free_rate: returns on risk-free assets (assumed constant until expiry)
        volatility: volatility of the underlying asset (standard deviation of log returns)
        num_simulations: number of simulated random price paths
        """
        # Brownian process parameters
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = maturity_days / 365
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

        # Simulation parameters
        self.num_simulations = num_simulations
        self.num_steps = maturity_days
        self.dt = self.time_to_maturity / self.num_steps

    def simulate_price_paths(self):
        """
        Simulates price movements for the underlying asset using a Brownian motion stochastic process.
        """
        np.random.seed(20)
        self.simulation_results = None

        # Initialize price movements array: rows are time steps, columns are different simulations
        price_paths = np.zeros((self.num_steps, self.num_simulations))        
        # Set the starting price for all simulations
        price_paths[0] = self.spot_price

        for t in range(1, self.num_steps):
            # Generate random values for Brownian motion (Gaussian distribution)
            random_values = np.random.standard_normal(self.num_simulations)
            # Update price at the next time step
            price_paths[t] = price_paths[t - 1] * np.exp((self.risk_free_rate - 0.5 * self.volatility ** 2) * self.dt + 
                                                         (self.volatility * np.sqrt(self.dt) * random_values))

        self.simulation_results = price_paths

    def _compute_call_option_price(self): 
        """
        Computes the price of a call option using simulated price paths.
        Payoff for a call option: max(S_T - K, 0), where S_T is the price at expiry.
        """
        if self.simulation_results is None:
            return -1
        payoffs = np.maximum(self.simulation_results[-1] - self.strike_price, 0)
        return np.exp(-self.risk_free_rate * self.time_to_maturity) * np.mean(payoffs)

    def _compute_put_option_price(self): 
        """
        Computes the price of a put option using simulated price paths.
        Payoff for a put option: max(K - S_T, 0), where S_T is the price at expiry.
        """
        if self.simulation_results is None:
            return -1
        payoffs = np.maximum(self.strike_price - self.simulation_results[-1], 0)
        return np.exp(-self.risk_free_rate * self.time_to_maturity) * np.mean(payoffs)
       
    def plot_simulation_paths(self, num_paths):
        """Plots the specified number of simulated price paths."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.simulation_results[:, :num_paths])
        plt.axhline(self.strike_price, color='black', label='Strike Price')
        plt.xlim([0, self.num_steps])
        plt.ylabel('Simulated Price Movements')
        plt.xlabel('Days to Maturity')
        plt.title(f'First {num_paths}/{self.num_simulations} Simulated Price Movements')
        plt.legend(loc='best')
        plt.show()
