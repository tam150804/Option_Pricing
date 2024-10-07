# Third party imports
import numpy as np
from scipy.stats import norm 

# Local package imports
from .base import PricingModel


class BSM(PricingModel):
    """ 
    Class implementing calculation for European option price using the Black-Scholes Formula.

    Call/Put option price is calculated with the following assumptions:
    - European option can only be exercised on the maturity date.
    - The underlying stock does not pay dividends during the option's lifetime.  
    - The risk-free rate and volatility are constant.
    - The Efficient Market Hypothesis holds — market movements cannot be predicted.
    - The underlying returns follow a lognormal distribution.
    """

    def __init__(self, spot_price, strike_price, maturity_days, risk_free_rate, volatility):
        """
        Initializes variables used in the Black-Scholes formula.

        spot_price: current stock or other underlying spot price
        strike_price: strike price for the option contract
        maturity_days: option contract maturity/exercise date in days
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        volatility: volatility of the underlying asset (standard deviation of asset's log returns)
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = maturity_days / 365
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def _compute_call_price(self): 
        """
        Computes the price for a call option according to the Black-Scholes formula.        
        Formula: S*N(d1) - PV(K)*N(d2)
        """
        # Calculate d1 and d2 using the Black-Scholes formula
        d1 = (np.log(self.spot_price / self.strike_price) + 
             (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
        
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        
        # Compute call option price
        call_price = (self.spot_price * norm.cdf(d1) - 
                      self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(d2))
        
        return call_price

    def _compute_put_price(self): 
        """
        Computes the price for a put option according to the Black-Scholes formula.        
        Formula: PV(K)*N(-d2) - S*N(-d1)
        """  
        # Calculate d1 and d2 using the Black-Scholes formula
        d1 = (np.log(self.spot_price / self.strike_price) + 
             (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))

        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        
        # Compute put option price
        put_price = (self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-d2) - 
                     self.spot_price * norm.cdf(-d1))
        
        return put_price
