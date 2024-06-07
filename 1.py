import numpy as np
import matplotlib.pyplot as plt

# Generate random housing dataset
np.random.seed(0)
num_samples = 1000
area = np.random.randint(500, 5000, num_samples)
bedrooms = np.random.randint(1, 6, num_samples)
age = np.random.randint(1, 50, num_samples)
price = 50000 + (area * 100) + (bedrooms * 20000) - (age * 500)

# Feature matrix X and target vector y
X = np.column_stack((area, bedrooms, age))
y = price.reshape(-1, 1)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term to X
X = np.column_stack((np.ones(num_samples), X))

# Linear regression using vectorization
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    costs = []
    for _ in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (1 / m) * alpha * (X.T.dot(errors))
        cost = compute_cost(X, y, theta)
        costs.append(cost)
    return theta, costs

# Initialize theta and hyperparameters
theta = np.zeros((X.shape[1], 1))
alpha = 0.01
num_iterations = 1000

# Run gradient descent
theta, costs = gradient_descent(X, y, theta, alpha, num_iterations)

# Plotting the cost curve
plt.plot(range(num_iterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.show()

# Print the learned parameters
print('Learned parameters:')
print('Intercept:', theta[0])
print('Coefficients:', theta[1:])

# Predicting house prices for new data
new_data = np.array([[3000, 4, 10]])  # New house data: area=3000, bedrooms=4, age=10
normalized_new_data = (new_data - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
normalized_new_data = np.column_stack((np.ones(1), normalized_new_data))
predicted_price = normalized_new_data.dot(theta)
print('Predicted price for new data:', predicted_price[0, 0])
