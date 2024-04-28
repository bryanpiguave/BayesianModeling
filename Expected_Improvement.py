import numpy as np 
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

"""
    This script demonstrates how to use the expected improvement acquisition 
    function to optimize a black-box function using a Gaussian Process.
"""


def black_box(x: np.ndarray):
    return np.sin(x) + np.cos(2*x)+ np.random.normal(0, 3)

def expected_improvement(x, gp_model:GaussianProcessRegressor, best_y:float)->np.ndarray:
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

x_range = np.linspace(0, 2*np.pi, 1000)
black_box_values = black_box(x_range)
num_samples=10  
x_samples = np.random.uniform(0, 2*np.pi, num_samples)
x_samples = np.sort(x_samples)
y_samples = black_box(x_samples)



kernel=RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(x_samples[:, np.newaxis], y_samples)
y_pred, sigma = gp.predict(x_range[:, np.newaxis], return_std=True)

plt.figure()
plt.scatter(x=x_samples, y=y_samples, c='red', label='Observations')
plt.plot(x_range,y_pred, 'b-', label='Prediction')
plt.fill_between(x_range, y_pred-1.96*sigma, y_pred+1.96*sigma, alpha=0.2, label='95% confidence interval')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




# Determine the point with the highest observed function value
best_idx = np.argmax(y_samples)
best_x = x_samples[best_idx]
best_y = y_samples[best_idx]

ei = expected_improvement(x_range, gp, best_y)

# Plot the expected improvement
plt.figure(figsize=(10, 6))
plt.plot(x_range, ei, color='green', label='Expected Improvement')
plt.xlabel('x')
plt.ylabel('Expected Improvement')
plt.title('Expected Improvement')
plt.legend()
plt.show()


