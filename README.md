# AI-Projects

## Introduction
Hi there! Welcome to my AI Projects repository. Here, you'll find a collection of my academic and personal projects that explore various facets of artificial intelligence and machine learning. These projects have been instrumental in my journey to understanding and applying AI concepts, from foundational algorithms to advanced neural network architectures.

Each project is meticulously documented to showcase the problem-solving approaches, implementation details, and the key insights gained. Whether it's probabilistic modeling, dimensionality reduction, clustering, regression, or reinforcement learning, this repository reflects my commitment to mastering the diverse landscape of AI.

Feel free to explore the projects and delve into the code. I hope you find them as exciting and enlightening as I did while working on them. If you have any questions or feedback, don't hesitate to reach out. Thank you for visiting!

## 1) Probabilistic Language Identification

### Description
This project involves developing a probabilistic model to identify the language of a given text. The model utilizes a Naive Bayes classifier trained on a dataset comprising text samples from various languages. The process includes data preprocessing, training the classifier, and evaluating its performance on language detection tasks.

### Key Components
- **Data Preprocessing**:
  - Tokenization: Splitting the text into tokens (words or characters).
  - Normalization: Converting text to a consistent format (e.g., lowercasing, removing punctuation).

- **Naive Bayes Classifier**:
  - Implementation of the Naive Bayes algorithm to calculate the probability of each language given the input text.
  - Training the classifier using a multilingual dataset.

- **Language Detection**:
  - Predicting the language of new text samples based on the highest probability calculated by the Naive Bayes classifier.
  - Evaluating the model's performance using metrics such as accuracy, precision, and recall.

## 2) Principal Component Analysis (PCA)

### Description
This project involves implementing Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset. The goal is to transform the data into a new coordinate system where the greatest variances are projected onto the first few principal components.

### Key Components
- **Data Standardization**:
  - Scaling the data to have zero mean and unit variance.

- **Covariance Matrix Calculation**:
  - Computing the covariance matrix to understand the relationships between different dimensions in the data.

- **Eigenvalue and Eigenvector Computation**:
  - Calculating eigenvalues and eigenvectors of the covariance matrix to identify the principal components.

- **Dimensionality Reduction**:
  - Projecting the original data onto the principal components to reduce the number of dimensions while retaining most of the variance.

## 3) Hierarchical Clustering

### Description
This project involves performing hierarchical clustering on a dataset to group similar data points into clusters. The approach used is agglomerative clustering, where each data point starts as its own cluster and pairs of clusters are merged iteratively based on their similarity.

### Key Components
- **Distance Metrics**:
  - Calculation of distances between data points using metrics like Euclidean, Manhattan, or cosine distance.

- **Linkage Criteria**:
  - Methods to determine the distance between clusters, including single linkage, complete linkage, and average linkage.

- **Dendrogram Construction**:
  - Building a dendrogram to visualize the hierarchical relationships between clusters and determine the optimal number of clusters.

## 4) Linear Regression

### Description
This project focuses on implementing linear regression to predict a target variable based on one or more predictor variables. The model is trained using a dataset, and its performance is evaluated on unseen data.

### Key Components
- **Data Visualization**:
  - Plotting data to understand the relationships between variables and identify patterns.

- **Model Training**:
  - Fitting a linear regression model to the training data using methods like ordinary least squares (OLS).

- **Prediction and Evaluation**:
  - Using the trained model to make predictions on new data.
  - Evaluating model performance using metrics such as mean squared error (MSE) and R-squared.

## 5) Introduction to PyTorch

### Description
This project introduces the PyTorch library for deep learning. It involves building and training neural networks using PyTorch and understanding its key features and capabilities.

### Key Components
- **PyTorch Basics**:
  - Introduction to tensors, operations, and automatic differentiation.

- **Neural Network Implementation**:
  - Building a simple neural network using PyTorch's `nn` module.

- **Training and Evaluation**:
  - Training the neural network on a dataset.
  - Evaluating the model's performance on test data.

## 6) Convolutional Neural Networks (LeNet)

### Description
This project involves implementing and training a Convolutional Neural Network (CNN) based on the LeNet architecture. The goal is to classify images from a dataset, such as MNIST, using deep learning techniques.

### Key Components
- **CNN Architecture**:
  - Building the LeNet architecture with convolutional, pooling, and fully connected layers.

- **Data Augmentation**:
  - Applying data augmentation techniques to increase the diversity of the training data.

- **Model Training and Evaluation**:
  - Training the CNN on the training dataset.
  - Evaluating the model's accuracy on the validation and test datasets.

## 7) A* Search Algorithm

### Description
This project involves implementing the A* search algorithm to solve pathfinding problems, such as navigating through a maze. The A* algorithm is known for its efficiency in finding the shortest path in a weighted graph.

### Key Components
- **Heuristic Function**:
  - Designing an admissible heuristic to estimate the cost from the current node to the goal.

- **Priority Queue**:
  - Using a priority queue to manage the exploration of nodes based on their estimated total cost.

- **Pathfinding and Optimization**:
  - Implementing the A* algorithm to find the optimal path from the start to the goal node.
  - Optimizing the algorithm for performance.

## 8) Minimax Algorithm for Teeko

### Description
This project involves developing an AI player for the game Teeko using the Minimax algorithm. The goal is to create a competitive AI that can play the game against human opponents or other AI players.

### Key Components
- **Game State Representation**:
  - Defining the game board and representing the state of the game.

- **Minimax Algorithm**:
  - Implementing the Minimax algorithm to evaluate possible moves and select the best one.
  - Incorporating alpha-beta pruning to optimize the algorithm's performance.

- **AI Player Development**:
  - Integrating the Minimax algorithm into the AI player.
  - Testing the AI against various opponents to assess its effectiveness.

## 9) Q-Learning for Reinforcement Learning

### Description
This project focuses on implementing the Q-Learning algorithm for reinforcement learning tasks. The goal is to train an agent to learn optimal policies for decision-making in an environment.

### Key Components
- **Environment Setup**:
  - Defining the environment in which the agent operates, including states, actions, and rewards.

- **Q-Learning Algorithm**:
  - Implementing the Q-Learning algorithm to update the Q-values based on the agent's experiences.

- **Policy Optimization**:
  - Training the agent to learn the optimal policy through exploration and exploitation.
  - Evaluating the agent's performance on various tasks.
