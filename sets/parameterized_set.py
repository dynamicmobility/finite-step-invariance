import numpy as np
from abc import ABC, abstractmethod

# Abstract class representing all types of parameterized sets
class ParameterizedSet(ABC):
    """Sets using row-wise (N, dimension) data and samples, and (N,) masks."""

    def __init__(self, dimension):
        self.dimension = dimension

    @abstractmethod
    def fit(self, data):
        pass

    # Samples N points  
    @abstractmethod
    def sample_from(self, N: int, rng: np.random.Generator | None = None) -> np.ndarray:
        pass

    @abstractmethod
    def contains_points(self, pts: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def load_from_row(self, row: np.ndarray):
        pass

    @abstractmethod
    def plot(self, ax, diml = 0, dimh = 1, alpha = 1.0, color = 'green', **kwargs):
        pass