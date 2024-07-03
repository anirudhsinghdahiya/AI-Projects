import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

def Q1():
    """
    Function Q1 reads the 'toy.csv' file into a pandas dataframe, cleans and processes the data by:
    - Dropping the 'Five-year running average' column if it exists.
    - Renaming the 'Category' and 'Annual number of days' columns to 'year' and 'days', respectively.
    - Stripping the 'year' column of any characters following a hyphen to retain only the year.
    - Saving the cleaned dataframe to 'hw5.csv' for further use.
    """
    df = pd.read_csv('toy.csv')
    
    # Check for and drop the 'Five-year running average' column.
    if 'Five-year running average' in df.columns:
        df.drop('Five-year running average', axis=1, inplace=True)
    
    # Rename the columns for consistency with expected output.
    df.rename(columns={'Category': 'year', 'Annual number of days': 'days'}, inplace=True)

    # Extract the year from a string if it's in a 'YYYY-YYYY' format.
    df['year'] = df['year'].astype(str).apply(lambda x: x.split('-')[0].strip())

    # Save the cleaned data to a new CSV file for subsequent processing.
    df.to_csv('hw5.csv', index=False)

def Q2():
    """
    Function Q2 plots the year against the number of frozen days. It reads the cleaned data from 'hw5.csv',
    creates a line plot with markers, labels the axes, adds a title, and saves the plot as 'plot.jpg'.
    """
    # Read the cleaned data for plotting.
    df = pd.read_csv('hw5.csv')
    
    # Set up the plot with a title and labels.
    plt.figure(figsize=(8, 6))
    plt.plot(df['year'], df['days'], marker='o', linestyle='-', color='cornflowerblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.title('Year vs Frozen Days')
    
    # Save the figure as a JPEG image.
    plt.savefig('plot.jpg')

# The remaining functions from Q3a to Q6 are steps in a linear regression analysis.
# Each function corresponds to a specific step in the analysis and uses matrix operations for computation.

def Q3a():
    """
    Function Q3a constructs the X matrix required for linear regression. The first column is all ones (bias term),
    and the second column contains the years. This function assumes the year data is already in integer format.
    The X matrix is printed and returned for use in the subsequent steps of the regression analysis.
    """
    # Read the processed data for creating the X matrix.
    df = pd.read_csv('hw5.csv')
    
    # Create the X matrix with a column of ones and a column of years.
    X = np.array([[1, x] for x in df['year']], dtype=np.int64)
    
    # Print and return the X matrix.
    print("Q3a:")
    print(X)
    return X

def Q3b():
    """
    Function Q3b creates the Y vector (matrix) for linear regression. This vector contains the dependent variable,
    which in this context is the number of frozen days. The Y vector is printed and returned for use in regression.
    """
    # Read the processed data for creating the Y vector.
    df = pd.read_csv('hw5.csv')
    
    # Create the Y vector from the 'days' column.
    Y = np.array(df['days'], dtype=np.int64)
    
    # Print and return the Y vector.
    print("Q3b:")
    print(Y)
    return Y

def Q3c(X):
    """
    Function Q3c calculates the matrix Z as the dot product of the transpose of X and X itself.
    This is part of the normal equation for computing the least squares solution to a linear regression.
    """
    # Calculate the dot product of X's transpose and X.
    Z = np.dot(X.T, X)
    
    # Convert Z to integer type for consistency in output and print.
    Z = Z.astype(np.int64)
    print("Q3c:")
    print(Z)
    return Z

def Q3d(Z):
    """
    Function Q3d computes the inverse of matrix Z. In the context of linear regression, this is used to compute
    the pseudo-inverse of the X matrix. The inversion is printed and returned.
    """
    # Convert Z to float for the inversion operation.
    Z = Z.astype(np.float64)
    
    # Compute the inverse of Z.
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)
    return I


def Q3e(I, X):
    """
    Function Q3e computes the pseudo-inverse of matrix X, denoted as PI. 
    This is done by taking the dot product of the inverse of Z (I) and the transpose of X.
    The pseudo-inverse is used to find the coefficients (B_hat) in the linear regression equation.
    """
    # Convert X to float for matrix multiplication.
    X = X.astype(np.float64)
    
    # Compute the pseudo-inverse of X.
    PI = np.dot(I, X.T)
    
    # Print and return the pseudo-inverse.
    print("Q3e:")
    print(PI)
    return PI

def Q3f(PI, Y):
    """
    Function Q3f computes the coefficients (B_hat) for the linear regression model by taking the dot product
    of the pseudo-inverse (PI) of X and the vector Y containing the target values.
    """
    # Compute the coefficients (B_hat) using the pseudo-inverse.
    B_hat = np.dot(PI, Y)
    
    # Print and return the coefficients.
    print("Q3f:")
    print(B_hat)
    return B_hat

def Q4(b_hat):
    """
    Function Q4 calculates the predicted number of frozen days for the year 2022 using the obtained coefficients.
    It applies the linear regression equation: y = b0 + b1 * x, where x is the year (2022 in this case).
    """
    # Calculate the predicted number of frozen days for the year 2022.
    x_test = 2022
    y_test = b_hat[0] + b_hat[1] * x_test
    
    # Print the result.
    print("Q4: " + str(y_test))

def Q5(b_hat):
    """
    Function Q5 examines the sign of the coefficient b1 (slope) to infer the trend in the number of frozen days.
    A positive coefficient indicates an increasing trend, negative indicates a decreasing trend, and zero indicates no change.
    """
    if b_hat[1] > 0:
        print("Q5a: >")
    elif b_hat[1] < 0:
        print("Q5a: <")
    else:
        print("Q5a: =")
    
    print("Q5b: " + "If the sign is positive, it indicates that our model anticipates a yearly increase in the number of days Lake Mendota was covered by ice. Conversely, a negative sign suggests a projected decrease in these days annually. If the sign is zero, our model foresees no change in the number of ice-covered days in the future; it remains constant.")

def Q6(b_hat):
    """
    Function Q6 calculates the year when the number of frozen days would be zero based on the obtained coefficients.
    It applies the linear regression equation with Y=0 to solve for the year (x). 
    """
    # Calculate the year when the number of frozen days would be zero.
    x = -b_hat[0] / b_hat[1]
    
    # Print the result along with an interpretation.
    print("Q6a: " + str(x))
    print("Q6b: " + "The year identified as x corresponds to 2455. Given that the coefficient of x in the model is negative, it indicates a declining or negative trend in the annual count of days Lake Mendota is covered by ice. This observation aligns with our findings, suggesting that according to our model, the number of frozen days is expected to diminish each year due to the negative trend projected into the future.")

# The main function orchestrates the execution of all defined tasks in sequence.
# It first cleans the data, then performs linear regression analysis, and finally, interprets and reports the results.
# It calls each function sequentially to execute the analysis.

def main():
    Q1()  # Clean and preprocess the data.
    Q2()  # Plot the cleaned data.
    X = Q3a()  # Construct the X matrix.
    Y = Q3b()  # Construct the Y vector.
    Z = Q3c(X)  # Compute the matrix Z.
    I = Q3d(Z)  # Compute the inverse of Z.
    PI = Q3e(I, X)  # Compute the pseudo-inverse of X.
    B_hat = Q3f(PI, Y)  # Compute the coefficients.
    y_test = Q4(B_hat)  # Predict the number of frozen days for 2022.
    Q5(B_hat)  # Analyze the trend based on the slope.
    x_0 = Q6(B_hat)  # Calculate the year when frozen days reach zero.

if __name__ == "__main__":
    main()
