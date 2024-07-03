import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Loads data from a CSV file and returns a list of dictionaries
def load_data(filepath):
    data = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(dict(row))  # Convert each row to a dictionary and add to the list
    return data

# Converts a dictionary representing a country to a feature vector
def calc_features(row):
    # Extracts and converts each required field to float
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    # Returns a NumPy array containing the features
    return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

# Performs Hierarchical Agglomerative Clustering (HAC)
def hac(features):
    features = np.array(features)  # Converts the list of features to a NumPy array for efficient calculations
    n = len(features)  # Number of data points
    clusters = {i: [i] for i in range(n)}  # Initializes clusters with each data point in its own cluster
    Z = np.zeros((n - 1, 4))  # Initializes the linkage matrix
    cluster_distances = {}  # Initializes a dictionary to store distances between clusters

    # Computes initial distances between all pairs of data points
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(features[i] - features[j])  # Euclidean distance
            cluster_distances[(i, j)] = cluster_distances[(j, i)] = dist  # Symmetric distances

    new_cluster_id = n  # Initializes the next cluster ID (after the original data points)

    # Main loop to iteratively merge clusters
    for k in range(n - 1):
        # Finds the pair of clusters with the minimum distance
        (x, y), min_dist = min(cluster_distances.items(), key=lambda item: item[1])
        # Updates the linkage matrix for the current merge
        Z[k, :2] = sorted([x, y])  # Cluster indices
        Z[k, 2] = min_dist  # Distance
        Z[k, 3] = len(clusters[x]) + len(clusters[y])  # Size of the new cluster

        # Merges the clusters
        clusters[new_cluster_id] = clusters[x] + clusters[y]
        del clusters[x], clusters[y]  # Removes the old clusters

        # Updates distances for the new cluster
        for i in clusters:
            if i != x and i != y:
                dist_x_i = cluster_distances.get((x, i), np.inf)
                dist_y_i = cluster_distances.get((y, i), np.inf)
                cluster_distances[(new_cluster_id, i)] = cluster_distances[(i, new_cluster_id)] = max(dist_x_i, dist_y_i)

        # Removes outdated distances involving the old clusters
        for key in list(cluster_distances.keys()):
            if x in key or y in key:
                del cluster_distances[key]

        new_cluster_id += 1  # Increments the cluster ID for the next new cluster

    return Z  # Returns the linkage matrix

# Generates a dendrogram from the clustering results
def fig_hac(Z, names):
    fig = plt.figure()  # Creates a new figure
    dendrogram(Z, labels=names, leaf_rotation=90)  # Plots the dendrogram
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    return fig  # Returns the figure object

# Normalizes the feature vectors
def normalize_features(features):
    features_array = np.array(features)  # Converts the list of features to a NumPy array
    means = features_array.mean(axis=0)  # Calculates the mean of each feature
    stds = features_array.std(axis=0)  # Calculates the standard deviation of each feature
    # Normalizes the features and returns them
    return ((features_array - means) / stds).tolist()

# Main function to orchestrate the loading, processing, and plotting of data
def main():
    data = load_data("countries.csv")  # Loads the data
    country_names = [row["Country"] for row in data]  # Extracts country names
    features = [calc_features(row) for row in data]  # Calculates feature vectors for each country
    features_normalized = normalize_features(features)  # Normalizes the feature vectors

    n = 50  # Sets the number of countries to include in the clustering

    Z_raw = hac(features[:n])  # Performs HAC on the raw features
    Z_normalized = hac(features_normalized[:n])  # Performs HAC on the normalized features

    fig_raw = fig_hac(Z_raw, country_names[:n])  # Generates a dendrogram for the raw clustering
    fig_normalized = fig_hac(Z_normalized, country_names[:n])  # Generates a dendrogram for the normalized clustering

    fig_raw.show()  # Displays the dendrogram for the raw clustering
    fig_normalized.show()  # Displays the dendrogram for the normalized clustering

if __name__ == "__main__":
    main()  # Executes the main function when the script is run directly
