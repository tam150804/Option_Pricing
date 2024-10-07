# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
from .base import PricingModel


class Binomial(PricingModel):
    """ 
    Class implementing calculation for European option price using BOPM (Binomial Option Pricing Model).
    It calculates option prices in discrete time (lattice-based), in specified number of time points between date of valuation and exercise date.
    This pricing model has three steps:
    - Price tree generation
    - Calculation of option value at each final node 
    - Sequential calculation of the option value at each preceding node
    """

    def __init__(self, spot_price, strike_price, time_to_maturity_days, risk_free_rate, volatility, steps):
        """
        Initializes variables used in Binomial Tree Model.

        spot_price: current stock or other underlying spot price
        strike_price: strike price for option contract
        time_to_maturity_days: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        volatility: volatility of the underlying asset (standard deviation of asset's log returns)
        steps: number of time periods between the valuation date and exercise date
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity_days / 365
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.steps = steps

    def _compute_call_price(self): 
        """Computes price for call option according to the Binomial formula."""
        # Time step, up and down factors
        delta_t = self.time_to_maturity / self.steps                             
        up_factor = np.exp(self.volatility * np.sqrt(delta_t))                 
        down_factor = 1.0 / up_factor                                    

        # Price vector initialization
        option_prices = np.zeros(self.steps + 1)                       

        # Underlying asset prices at different time points
        final_asset_prices = np.array( [(self.spot_price * up_factor**j * down_factor**(self.steps - j)) for j in range(self.steps + 1)])

        risk_free_compound = np.exp(self.risk_free_rate * delta_t)                  # risk-free compounded return
        prob_up = (risk_free_compound - down_factor) / (up_factor - down_factor)    # risk-neutral up probability
        prob_down = 1.0 - prob_up                                                   # risk-neutral down probability   

        option_prices[:] = np.maximum(final_asset_prices - self.strike_price, 0.0)
    
        # Option price back-calculation
        for i in range(self.steps - 1, -1, -1):
            option_prices[:-1] = np.exp(-self.risk_free_rate * delta_t) * (prob_up * option_prices[1:] + prob_down * option_prices[:-1]) 

        return option_prices[0]

    def _compute_put_price(self): 
        """Computes price for put option according to the Binomial formula."""  
        # Time step, up and down factors
        delta_t = self.time_to_maturity / self.steps                             
        up_factor = np.exp(self.volatility * np.sqrt(delta_t))                 
        down_factor = 1.0 / up_factor                                    

        # Price vector initialization
        option_prices = np.zeros(self.steps + 1)                       

        # Underlying asset prices at different time points
        final_asset_prices = np.array( [(self.spot_price * up_factor**j * down_factor**(self.steps - j)) for j in range(self.steps + 1)])

        risk_free_compound = np.exp(self.risk_free_rate * delta_t)                  # risk-free compounded return
        prob_up = (risk_free_compound - down_factor) / (up_factor - down_factor)    # risk-neutral up probability
        prob_down = 1.0 - prob_up                                                   # risk-neutral down probability   

        option_prices[:] = np.maximum(self.strike_price - final_asset_prices, 0.0)
    
        # Option price back-calculation
        for i in range(self.steps - 1, -1, -1):
            option_prices[:-1] = np.exp(-self.risk_free_rate * delta_t) * (prob_up * option_prices[1:] + prob_down * option_prices[:-1]) 

        return option_prices[0]
