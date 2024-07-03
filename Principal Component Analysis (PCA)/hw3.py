from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Load the dataset from the specified file
    x = np.load(filename)
    # Calculate the mean of the dataset along each feature (column)
    mu_x = np.mean(x, axis=0)
    # Subtract the mean from the dataset to center it
    x_cent = x - mu_x
    return x_cent
    

def get_covariance(dataset):
    # Calculate the covariance matrix of the centered dataset
    # by multiplying the dataset transpose by the dataset itself
    # and normalizing by the number of samples minus one
    S = np.dot(dataset.T, dataset) / (len(dataset) - 1)
    return S

def get_eig(S, m):
    # Compute all eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = eigh(S)
    
    # Sort the eigenvalues in descending order and get the corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx][:m]
    sorted_eigenvectors = eigenvectors[:, idx][:, :m]
    
    # Normalize each eigenvector to have a Euclidean norm of 1
    normalized_eigenvectors = sorted_eigenvectors / np.linalg.norm(sorted_eigenvectors, axis=0)
    
    # Create a diagonal matrix from the largest 'm' eigenvalues
    Lambda = np.diag(sorted_eigenvalues)
    
    return Lambda, normalized_eigenvectors

def get_eig_prop(S, prop):
    # Compute all eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = eigh(S)
    # Sort the eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Normalize each eigenvector to have a Euclidean norm of 1
    normalized_eigenvectors = sorted_eigenvectors / np.linalg.norm(sorted_eigenvectors, axis=0)

    # Calculate the cumulative sum of the sorted eigenvalues and normalize by the total sum
    # to find the proportion of variance explained by each eigenvalue
    cumsum_eigenvalues = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    # Find the number of eigenvalues needed to explain at least the given proportion of variance
    num_eigenvalues_to_keep = np.argmax(cumsum_eigenvalues >= prop) + 1

    # Create a diagonal matrix from the necessary number of eigenvalues
    Lambda = np.diag(sorted_eigenvalues[:num_eigenvalues_to_keep])
    # Select the corresponding eigenvectors
    U = normalized_eigenvectors[:, :num_eigenvalues_to_keep]

    return Lambda, U


def project_image(image, U):
    # Project the given image onto the subspace defined by the eigenvectors in U
    # to get the projection weights (coordinates in the new subspace)
    weights = U.T @ image
    
    # Reconstruct the image from the projection weights and the eigenvectors
    # to obtain the approximation of the original image in the original space
    reconstruction = U @ weights
    
    return reconstruction  # Return the reconstructed d x 1 numpy array


def display_image(orig, proj):
    # Reshape the original and projected images to 64x64 for display
    orig_reshaped = orig.reshape((64, 64))
    proj_reshaped = proj.reshape((64, 64))
    
    # Create a figure with two subplots side by side for the original and projected images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    
    # Display the original image in the first subplot
    ax1.imshow(orig_reshaped, aspect='equal')
    ax1.set_title('Original')
    ax1.axis('off')  # Hide axes

    # Display the projected (reconstructed) image in the second subplot
    ax2.imshow(proj_reshaped, aspect='equal')
    ax2.set_title('Projection')
    ax2.axis('off')  # Hide axes

    # Add colorbars next to each image to indicate intensity levels
    plt.colorbar(ax1.imshow(orig_reshaped, aspect='equal'), ax=ax1)
    plt.colorbar(ax2.imshow(proj_reshaped, aspect='equal'), ax=ax2)

    return fig, ax1, ax2

