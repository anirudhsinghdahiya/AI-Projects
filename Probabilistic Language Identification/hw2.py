import sys
import math
import string   

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup
    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return (e, s)

def shred(filename):
    '''
    Counts occurrences of each letter in the specified file.

    Args:
        filename: The name of the file to be processed.

    Returns:
        A dictionary with letters as keys and their counts as values.
    '''
    # Initialize a dictionary for letter counts, with each letter set to 0
    letter_counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    
    # Read the file and increment count for each letter found
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            for char in line.upper():
                if char in letter_counts:
                    letter_counts[char] += 1

    return letter_counts

def compute_q1(letter_counts):
    '''
    Prints letter frequencies, labeled as 'Q1'.

    Args:
        letter_counts: A dictionary with letters and their counts.
    '''
    print("Q1")
    for letter in sorted(letter_counts.keys()):
        print(f"{letter} {letter_counts[letter]}")

def compute_q2(e, s, letter_counts):
    '''
    Calculates and prints the log probabilities for the first character, labeled as 'Q2'.

    Args:
        e: List of English letter probabilities.
        s: List of Spanish letter probabilities.
        letter_counts: A dictionary with letters and their counts.
    '''
    print("Q2")
    a_count = letter_counts.get('A', 0)
    q2_e = a_count * math.log(e[0]) if a_count > 0 else 0.0
    q2_s = a_count * math.log(s[0]) if a_count > 0 else 0.0
    print(f"{q2_e:.4f}")
    print(f"{q2_s:.4f}")

def compute_q3(e, s, letter_counts):
    '''
    Computes and prints the log likelihoods for English and Spanish texts, labeled as 'Q3'.

    Args:
        e: List of English letter probabilities.
        s: List of Spanish letter probabilities.
        letter_counts: A dictionary with letters and their counts.
    '''
    print("Q3")
    F_Eng = math.log(0.6)  # Prior probability for English
    F_Spa = math.log(0.4)  # Prior probability for Spanish
    for letter, count in letter_counts.items():
        index = ord(letter) - ord('A')
        F_Eng += count * math.log(e[index])
        F_Spa += count * math.log(s[index])
    print(f"{F_Eng:.4f}")
    print(f"{F_Spa:.4f}")
    return F_Eng, F_Spa

def compute_q4(F_Eng, F_Spa):
    '''
    Calculates and prints the posterior probability for the text being English, labeled as 'Q4'.

    Args:
        F_Eng: Log likelihood for English.
        F_Spa: Log likelihood for Spanish.
    '''
    print("Q4")
    if F_Spa - F_Eng > 100:
        P_Eng_X = 0.0
    elif F_Eng - F_Spa > 100:
        P_Eng_X = 1.0
    else:
        P_Eng_X = 1 / (1 + math.exp(F_Spa - F_Eng))
    print(f"{P_Eng_X:.4f}")

def main():
    '''
    Main function to orchestrate the computation and display of Q1 to Q4.
    '''
    e, s = get_parameter_vectors()
    letter_counts = shred("letter.txt")
    compute_q1(letter_counts)
    compute_q2(e, s, letter_counts)
    F_Eng, F_Spa = compute_q3(e, s, letter_counts)
    compute_q4(F_Eng, F_Spa)

if __name__ == "__main__":
    main()