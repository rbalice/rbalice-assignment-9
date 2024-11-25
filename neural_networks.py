import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        # self.W1 = np.random.randn(input_dim, hidden_dim) * 0.3
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # self.W2 = np.random.randn(hidden_dim, output_dim) * 0.3
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Store activations and gradients for visualization
        self.hidden_features = None
        self.gradients = {
            'W1': np.zeros_like(self.W1),
            'W2': np.zeros_like(self.W2)
        }
    
    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
    
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.hidden_features = self.a1  # Store for visualization
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.z2))
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        
        # Compute gradients
        # delta2 = (self.forward(X) - y) * (1 - np.tanh(self.z2)**2)
        delta2 = self.A2 - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    # y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    ## figure 1
    # Plot hidden features
    hidden_features = np.clip(mlp.hidden_features, -1, 1)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                     c=y.ravel(), cmap='bwr', alpha=0.7)
    
    # Simplified hyperplane in hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2) / (mlp.W2[2] + 1e-6)
    zz = np.clip(zz, -1, 1)
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    
    ax_hidden.set_title('Hidden Space at Step ' + str(frame * 10))
    ax_hidden.set_xlim([-1, 1])
    ax_hidden.set_ylim([-1, 1])
    ax_hidden.set_zlim([-1, 1])
    ax_hidden.view_init(elev=20, azim=45)  

    ## figure 2
    # Plot input space decision boundary with clearer separation
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(points)
    Z = Z.reshape(xx.shape)
    
    # Use binary threshold for clearer decision boundary
    ax_input.contourf(xx, yy, Z > 0.5, levels=1, colors=['blue', 'red'], alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')
    ax_input.set_title('Input Space at Step ' + str(frame * 10))
    ax_input.set_xlim([-3, 3])
    ax_input.set_ylim([-3, 3])

    ## figure 3
    # Visualize gradients
    ax_gradient.set_title('Gradients at Step ' + str(frame * 10))
    
    # Plot neurons
    neurons = {
        'x1': (0.0, 0.0), 'x2': (0.0, 1.0),  # Input neurons
        'h1': (0.5, 0.0), 'h2': (0.5, 0.5), 'h3': (0.5, 1.0),  # Hidden neurons
        'y': (1.0, 0.0)  # Output neuron
    }
    
    # Draw neurons as circles
    for name, pos in neurons.items():
        circle = Circle(pos, 0.05, color='blue')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0]-0.02, pos[1]+0.1, name)
    
    # Draw connections with gradient-based thickness
    max_gradient = max(np.max(np.abs(mlp.gradients['W1'])), np.max(np.abs(mlp.gradients['W2'])))
    
    # Input to hidden connections
    for i in range(mlp.W1.shape[0]):  
        for j in range(mlp.W1.shape[1]): 
            start = neurons[f'x{i+1}']
            end = neurons[f'h{j+1}']
            width = 10 * np.abs(mlp.gradients['W1'][i, j]) / (max_gradient + 1e-6)
            ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 
                            linewidth=width, color='purple', alpha=0.5)
    
    # Hidden to output connections
    for i in range(3):
        start = neurons[f'h{i+1}']
        end = neurons['y']
        # width = 5 * np.abs(mlp.gradients['W2'][i,0]) / (max_gradient + 1e-6)
        width = 10 * np.abs(mlp.gradients['W2'][i, 0]) / (max_gradient + 1e-6)
        ax_gradient.plot([start[0], end[0]], [start[1], end[1]], 
                        linewidth=width, color='purple', alpha=0.5)
    
    ax_gradient.set_xlim([-0.2, 1.2])
    ax_gradient.set_ylim([-0.2, 1.2])
    ax_gradient.axis('equal')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)