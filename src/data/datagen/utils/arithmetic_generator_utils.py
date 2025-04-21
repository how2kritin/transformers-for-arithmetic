import random
import numpy as np


def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def format_expression(a, b):
    """Format the expression string for the transformer input."""
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
    """Calculate the correct arithmetic result for a and b."""
    # Calculate the actual result
    return a + b  # This works for all cases since b is already negative for a-b or a is negative for -a+b


def convert_digits_to_number(digits):
    """Convert a list of digits to a number."""
    number = 0
    for digit in reversed(digits):
        number = number * 10 + digit
    return number


def get_random_digit_sequence(length, exclude_leading_zero=True):
    """Generate a random sequence of digits with improved distribution."""
    if length <= 0:
        return []

    # First digit (shouldn't be 0 unless explicitly allowed)
    if exclude_leading_zero:
        first_digit = random.randint(1, 9)
    else:
        first_digit = random.randint(0, 9)

    # Rest of the digits
    rest_digits = [random.randint(0, 9) for _ in range(length - 1)]

    return [first_digit] + rest_digits


def diversify_digit_generation():
    """Occasionally use a different distribution to increase diversity."""
    if random.random() < 0.2:  # 20% chance to use a different distribution
        # Use different distributions sometimes to increase diversity
        distributions = [
            lambda: random.randint(0, 9),  # Uniform distribution
            lambda: min(9, max(0, int(np.random.normal(5, 2)))),  # Normal distribution centered at 5
            lambda: min(9, max(0, int(np.random.exponential(3)))),  # Exponential distribution
            lambda: random.choice([0, 1, 9])  # Specific values that might be underrepresented
        ]
        return random.choice(distributions)()
    else:
        return random.randint(0, 9)  # Default uniform distribution