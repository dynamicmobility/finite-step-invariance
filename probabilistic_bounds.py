from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import beta 

# Binomial tail inversion to calculate accuracy level given a test datasest
def binomial_tail(misses: int, num_samples: int) -> float:
    """
    Compute the largest p such that:
        BinomialCDF(k=misses; n=num_samples, p) >= 1e-9
    """
    return 1- beta.ppf(0.000000001, num_samples-misses, misses+1)

if __name__ == '__main__':
    # Test cases
    print("test case :", binomial_tail(33, 2000))