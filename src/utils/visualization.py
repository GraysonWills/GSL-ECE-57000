import matplotlib.pyplot as plt
import numpy as np

def visualize_correlation_map(correlation_map):
    plt.imshow(correlation_map, cmap='viridis')
    plt.title("Correlation Map")
    plt.colorbar()
    plt.show()

def plot_trajectory(correlation_map):
    H, W = correlation_map.shape[:2]
    plt.quiver(range(W), range(H), correlation_map[..., 0], correlation_map[..., 1], scale=50)
    plt.title("Body Trajectory")
    plt.show()
