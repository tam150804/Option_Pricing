from enum import Enum
from abc import ABC, abstractmethod

class OptionType(Enum):
    CALL = 'Call Option'
    PUT = 'Put Option'

class PricingModel(ABC):
    """Abstract class defining interface for option pricing models."""

    def compute_option_price(self, option_type):
        """Computes call/put option price based on the specified parameter."""
        if option_type == OptionType.CALL.value:
            return self._compute_call_price()
        elif option_type == OptionType.PUT.value:
            return self._compute_put_price()
        else:
            return -1

    @abstractmethod
    def _compute_call_price(self):
        """Computes price for call option."""
        pass

    @abstractmethod
    def _compute_put_price(self):
        """Computes price for put option."""
        pass
