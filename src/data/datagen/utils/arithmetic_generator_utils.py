import random
import numpy as np


def set_random_seed(seed: int):
    random.seed(seed)
    try:
        np.random.seed(seed)
    except ImportError:
        pass


def format_expression(a: int, b: int):
    """format the expression string for the transformer input"""
    if a >= 0 and b >= 0:
        # both positive: a+b
        return f"{a}+{b}"
    elif a >= 0 and b < 0:
        # first positive, second negative: a-|b|
        return f"{a}-{abs(b)}"
    elif a < 0 and b >= 0:
        # first negative, second positive: -|a|+b
        return f"-{abs(a)}+{b}"
    else:
        # both negative: -|a|-|b|
        return f"-{abs(a)}-{abs(b)}"


def calculate_correct_result(a: int, b: int):
    """calculate the correct arithmetic result for a and b"""
    return a + b  # this works for all cases since b is already negative for a-b or a is negative for -a+b


def convert_digits_to_number(digits: list[int]):
    """convert a list of digits to a number."""
    number = 0
    for digit in reversed(digits):
        number = number * 10 + digit
    return number


def get_random_digit_sequence(length: int, exclude_leading_zero: bool = True):
    """generate a random sequence of digits."""
    if length <= 0:
        return []

    # first digit (shouldn't be 0 unless explicitly required)
    if exclude_leading_zero:
        first_digit = random.randint(1, 9)
    else:
        first_digit = random.randint(0, 9)

    # rest of the digits
    rest_digits = [random.randint(0, 9) for _ in range(length - 1)]

    return [first_digit] + rest_digits


def diversify_digit_generation():
    """occasionally use a different distribution to increase diversity."""
    if random.random() < 0.2:  # 20% chance to use a different distribution
        # use different distributions sometimes to increase diversity
        distributions = [
            lambda: random.randint(0, 9),  # uniform distribution
            lambda: min(9, max(0, int(np.random.normal(5, 2)))),  # normal distribution centered at 5
            lambda: min(9, max(0, int(np.random.exponential(3)))),  # exponential distribution
            lambda: random.choice([0, 1, 9])  # specific values that might be underrepresented (edge cases, basically)
        ]
        return random.choice(distributions)()
    else:
        return random.randint(0, 9)  # default uniform distribution