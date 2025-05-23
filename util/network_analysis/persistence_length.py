import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def compute_tangent_vectors(positions):
    """Compute the tangent vectors for the polymer chain."""
    tangent_vectors = []
    norms = []
    for i in range(len(positions) - 1):
        tangent_vector = positions[i + 1] - positions[i]
        norm = np.linalg.norm(tangent_vector)
        tangent_vectors.append(tangent_vector / norm)  # Normalize
        norms.append(norm)
    return np.array(tangent_vectors), np.array(norms)

# def compute_cosine_of_angles(tangent_vectors):
#     """Compute the cosine of the angles between consecutive tangent vectors."""
#     cos_angles = []
#     for i in range(len(tangent_vectors) - 1):
#         cos_angle = np.dot(tangent_vectors[i], tangent_vectors[i + 1])  # Dot product
#         cos_angles.append(cos_angle)
#     return np.array(cos_angles)

def compute_autocorrelation(cos_angles):
    """Compute the autocorrelation function of the cosine angles."""
    # autocorrelation = []
    # n = len(cos_angles)
    # for lag in range(n):
    #     autocorr = np.mean(cos_angles[:n - lag] * cos_angles[lag:])
        
    #     autocorrelation.append(autocorr)
    n = len(cos_angles) + 1
    inner = np.inner(cos_angles, cos_angles)
    autocorrelation = np.zeros(n - 1)
    for i in range(n-1):
        autocorrelation[:(n-1)-i] += inner[i, i:]
    
    norm = np.linspace(n-1, 1, n-1)
    autocorrelation = autocorrelation / norm
    
    return np.array(autocorrelation)

def fit_func(x, m):
    return m*x

def fit_exponential_decay(autocorrelation, plot = False):
    """Fit an exponential decay to the autocorrelation function."""
    # Use logarithmic fitting to extract persistence length
    time = np.arange(len(autocorrelation))
    log_autocorrelation = np.log(autocorrelation)
    valid = ~np.isnan(log_autocorrelation) 
    time = time[valid]
    log_autocorrelation = log_autocorrelation[valid]
    
    # Fit to a straight line: log(C(s)) = -s / persistence_length
    params = np.polyfit(time, log_autocorrelation, 1)
    
    persistence_length = -1 / params[0] 
    
    if plot:
        plt.figure()
        plt.plot(time, log_autocorrelation, label="Log of Autocorrelation")
        plt.xlabel('Lag (s)')
        plt.ylabel('Log(Cosine of angles)')
        plt.title(f"Fitted Exponential Decay, Persistence Length = {persistence_length:.2f}")
        plt.legend()
        plt.show()
        
        plt.plot(time, params[0]*time + params[1])
        
    return persistence_length, time, log_autocorrelation, params

def fit_pl_fibril(bond_dist, autocorr, plot = False, fix_yint = False):
    """Fit an exponential decay to the autocorrelation function."""
    # Use logarithmic fitting to extract persistence length
    time = bond_dist
    log_autocorrelation = np.log(autocorr)
    # Avoid taking log(0) by removing zero values
    valid = ~np.isnan(log_autocorrelation) 
    time = time[valid]
    log_autocorrelation = log_autocorrelation[valid]
    
    # Fit to a straight line: log(C(s)) = -s / persistence_length
    if fix_yint:
        params, _ =  curve_fit(fit_func, time, log_autocorrelation)
    else:
        params = np.polyfit(time, log_autocorrelation, 1)

    
    persistence_length = -1 / params[0]  # Extract the persistence length (negative of the slope)
    
    if plot:
        # Plot the autocorrelation function and the fitted exponential decay
        plt.figure()
        plt.scatter(time, log_autocorrelation, s = 1, color = 'k')
        plt.xlabel('Lag (s)')
        plt.ylabel('Log(Cosine of angles)')
        plt.title(f"Fitted Exponential Decay, Persistence Length = {persistence_length:.2f}")
        if fix_yint:
            plt.plot(time, params[0]*time, linestyle = '--', c= 'r' )
        else:
            plt.plot(time, params[0]*time + params[1], linestyle = '--', c = 'r')
        
    return persistence_length, time, log_autocorrelation, params
    
    
    

def do_persistence_length(positions, plot = False):
    tangent_vectors, bond_lengths = compute_tangent_vectors(positions)
    autocorrelation = compute_autocorrelation(tangent_vectors)
    
    persistence_length, time, log_autocorrelation, params = fit_exponential_decay(autocorrelation, plot)
    return persistence_length, tangent_vectors, autocorrelation, bond_lengths
    
    
def example():
    # Example input: 2D array with points in 3D (each row is a point, each column is x, y, z)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 0.0],
        [4.0, 3.0, 0.0],
        [5.0, 5.0, 0.0]
    ])

    do_persistence_length(positions, plot = True)
    
if __name__ == '__main__':
    example()

