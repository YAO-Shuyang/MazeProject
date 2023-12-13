import numpy as np
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    vec = poisson.rvs(6, size = 1000)
    
    noise = (norm.rvs(loc = 0, scale = 1, size = 1000)//1).astype(int)
    print(noise)
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    print(len(vec), len(vec+noise))
    axes[0].hist(vec, range=(0.5, np.max(vec)+0.5), bins = int(np.max(vec)))
    axes[1].hist(vec+noise, range=(0.5, np.max(vec+noise)+0.5), bins = int(np.max(vec+noise)))
    plt.show()