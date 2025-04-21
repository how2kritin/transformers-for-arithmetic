import random

def set_random_seed(seed):
    """set the random seed for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

def format_expression(a, b):
    """format the expression string for the transformer input."""
    if a >= 0 and b >= 0:
        # both positive: a+b
        return f"{a}+{b}"
    elif a >= 0 and b < 0:
        # first positive, second negative: a-|b|
        return f"{a}-{abs(b)}"
    elif a < 0 and b >= 0:
        # first negative, second positive: -|a|+b
        return f"{a}+{b}"
    else:
        # both negative: -|a|-|b|
        return f"{a}-{abs(b)}"

def calculate_correct_result(a, b):
    """calculate the correct arithmetic result for a and b."""
    # calculate the actual result
    return a + b  # this works for all cases since b is already negative for a-b or a is negative for -a+b

def convert_digits_to_number(digits):
    """convert a list of digits to a number."""
    number = 0
    for digit in reversed(digits):
        number = number * 10 + digit
    return number