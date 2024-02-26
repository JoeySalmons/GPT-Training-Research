"""
original: 5/5/23
current: 9/2/23
Generate and prepare data.
The data is generated by randomly generating problems and their solutions.
The problems are in a form like this: 123 + 456 = 579 for base 10, but the problems can be in any base.

numbers are reversed, so 123 represents the value of three hundred twenty one

Future work:
generate problems with operators switched with other symbols (as of 9/2 it only works with + and - for two numbers)
use PEMDOS - more operators and order of operations
use decimals
use other functions like log, sin, cos, tan, etc.
generate other, more complicated problems with variables, derivatives, integrals, summations, limits, etc.
"""

import tiktoken
import os
import numpy as np
import pickle
import torch
import random
import statistics
from scipy import stats
import string

# optimized code (~2x faster)
# needs to be optimized by pre-generating the data and loading it during training

def generate_random_number_opt(base_symbols, min_length, max_length):
    # Determine the length of the number within the specified range
    num_length = random.randint(min_length, max_length)

    # Generate random digits using base_symbols
    random_digits = [random.choice(base_symbols) for _ in range(num_length)]

    return ''.join(random_digits)


def perform_base_operation_opt(num1, num2, base_symbols, operator):
    base = len(base_symbols)

    max_len = max(len(num1), len(num2))
    result = []
    carry = 0

    for i in range(max_len):
        digit1 = base_symbols.index(num1[i]) if i < len(num1) else 0
        digit2 = base_symbols.index(num2[i]) if i < len(num2) else 0

        if operator == '+':
            # Addition
            total = digit1 + digit2 + carry
            carry = total // base
            remainder = total % base
        elif operator == '-':
            # Subtraction
            total = digit1 - digit2 - carry
            carry = 0 if total >= 0 else 1
            remainder = (total + base) % base
        # only do + and - for now
        else:
            raise ValueError("Invalid operator")

        result.append(base_symbols[remainder])

    # If there's a carry left, add it as the most significant digit
    if carry:
        result.append(base_symbols[carry])
    # f"{num1} {operator} {num2} = {result}"
    return ''.join(result)


def generate_problems(length, num_samples, base_symbols=string.ascii_lowercase, exact_length=False,
                          problem_operators=["+"], randomize=False):
    """
    Generate problems in the form of a string.
    length is the length of each sample of problems (equivalent to context length or block size)
    num_samples is the number of samples of problems to generate (equivalent to batch size)
    a sample is a string of length length containing problems
    a problem is a string of the form "num1 operator num2 = answer"
    base_symbols is a string containing the symbols to use for the base with length equal to the base
    exact_length is a boolean that determines whether the length of the sample is exactly length or less than or equal to length
    problem_operators is a list of the operators to use in the problems
    randomize is a boolean that determines whether the base_symbols are randomized for each problem set
    """
    base = len(base_symbols)
    operator = random.choice(problem_operators)
    max_length = 12  # maximum length of characters for each number in the problems
    min_length = 2
    samples = []

    if randomize:
        # shuffle the base symbols
        base_symbols = ''.join(random.sample(base_symbols, base))

    # generate problems until the length of all the problems in the sample is greater than or equal to length
    for i in range(num_samples):
        problems = []
        total_length = 0

#        if randomize:
#            # shuffle the base symbols
#            base_symbols = ''.join(random.sample(base_symbols, len(base_symbols)))

        # generate problems until the length of all the problems in the sample is greater than or equal to the desired length
        while total_length < length:
            num1 = generate_random_number_opt(base_symbols, min_length, max_length)
            num2 = generate_random_number_opt(base_symbols, min_length, max_length)

            result = perform_base_operation_opt(num1, num2, base_symbols, operator)
            prob_str = f"{num1} {operator} {num2} = {result}"

            problems.append(prob_str)
            problem_length = len(prob_str)
            total_length += problem_length

        # convert the list of problems to a string
        problems_str = '\n' + '\n'.join(problems)

        # length_of_problems_str = len(problems_str)
        # trim the string to the correct length
        problems_str = problems_str[:length]
        if not exact_length:
            # remove any incomplete problems at the end of the string
            # find the last \n
            last_newline_index = problems_str.rfind('\n')
            if last_newline_index != -1:
                problems_str = problems_str[:last_newline_index]

        samples.append(problems_str)

    return samples

# length is block_size
# num_samples is batch_size
def get_train_data(length, num_samples=1, base_symbols=string.ascii_lowercase, operators=['+'], out_dir=None,
                       randomize=False):
    """
    Get and prepare training data.
    """
    # get data
    data = generate_problems(length, num_samples, base_symbols, exact_length=True, problem_operators=operators,
                                 randomize=randomize)
    # data is a list of stings containing batch_size samples each of length block_size
    # print(f"generated {num_samples} samples of length {length}")
    # load metadata
    if out_dir is None:
        print("out_dir is None")
        out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, 'meta.pkl'), 'rb') as f:
        # print("loading meta.pkl")
        meta = pickle.load(f)

    stoi = meta['stoi']
    # encode text to integers
    ids = [[stoi[c] for c in text] for text in data]

    # print("the length of ids is:")
    # print(len(ids))
    # print("the type of ids is:")
    # print(type(ids))
    # print("the type of ids[0] is:")
    # print(type(ids[0]))
    # print("the type of ids[0][0] is:")
    # print(type(ids[0][0]))

    # convert to numpy array with int32 data type
    ids = np.array(ids, dtype=np.int32)
    # the above line gives
    # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1024,) + inhomogeneous part.

    # for testing, print out the vocab
    # print("the vocab is:")
    # print(meta['stoi'])

    return ids

# alternative int to base converter
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]
# function test
str_list = numberToBase(453464564, 26+26+10)
print(str_list)