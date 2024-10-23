import numpy as np
import matplotlib.pyplot as plt

def func(p, s):
    return np.clip(s * (0.2 * p**2 + 0.3 * p + 0.46) + (1-s) * (-0.2 * (1-p)**2 - 0.3 * (1-p) + 0.5), 1e-10, 1-1e-10)

class FeatureKnockOut:
    def __init__(self, p=0.2, p0=0.5):
        self._p = p
        self.p0 = p0
        print(f"p0: {p0}, p: {p}")
    
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        simu_seq = []
        
        for seq in sequences:
            p = [self.p0]
            simu = [1]  
            for i in range(len(seq) - 1):
                simu.append(np.random.choice([0, 1], p=[1 - p[-1], p[-1]]))
                p_new = np.random.choice(
                    [
                        func(p[-1], simu[-1]),
                        func(p[-1], 1-simu[-1])
                    ], 
                    p=[1 - self._p, self._p]
                )
                p.append(p_new)
                
            simu_seq.append(np.array(simu, np.int64))
            
            #plt.plot(p, marker='s')
            #plt.plot(simu_seq[-1][1:])
            #plt.show()

        return simu_seq

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.linspace(0, 1, 1000)
    y = func(x, 0.6)
    plt.plot(x, y)
    plt.axis([0, 1, 0, 1])
    plt.plot([0, 1], [0, 1], 'k--')
    y2 = func(x, 0.4)
    plt.plot(x, y2)
    plt.show()
    
    y1 = func(x, 1) * 0.01 + (1-func(x, 1)) * 0.99
    plt.plot([0, 1], [0, 1], 'k--')
    y2 = func(x, 0) * 0.01 + (1-func(x, 0)) * 0.99
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    
    y1 = func(x, 1) * 0.6 + (1-func(x, 1)) * 0.4
    y2 = func(x, 0) * 0.6 + (1-func(x, 0)) * 0.4
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    
    y1 = func(x, 1) * 0.4 + (1-func(x, 1)) * 0.6
    y2 = func(x, 0) * 0.4 + (1-func(x, 0)) * 0.6
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()