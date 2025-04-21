import random
from src.data.datagen.utils.arithmetic_generator_utils import convert_digits_to_number

def generate_no_carry_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """generate addition problems where no carrying is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    # generate individual digits ensuring no carrying
    a_digits = []
    b_digits = []

    for i in range(max(digits_a, digits_b)):
        if i < digits_a:
            # for digit positions in a
            if i < digits_b:
                # ensure sum with corresponding digit in b is less than 10
                a_digit = random.randint(0, 9)
                b_digit = random.randint(0, 9 - a_digit)
            else:
                # no corresponding digit in b, any digit is fine
                a_digit = random.randint(0, 9)
                b_digit = 0
        else:
            # position not in a
            a_digit = 0
            b_digit = random.randint(0, 9)

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits[:digits_a])
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result
    result = a + b

    return a, b, result


def generate_carry_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_carries=1):
    """generate addition problems with at least one carry operation."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    carries_placed = 0
    max_potential_carries = min(digits_a, digits_b)

    if min_carries > max_potential_carries:
        # can't satisfy the minimum carries requirement
        # adjust parameters to ensure we can have at least one carry
        digits_a = max(digits_a, min_carries)
        digits_b = max(digits_b, min_carries)
        max_potential_carries = min(digits_a, digits_b)

    # generate individual digits ensuring at least one carry
    a_digits = [0] * max(digits_a, digits_b)
    b_digits = [0] * max(digits_a, digits_b)

    # first, place required carries
    while carries_placed < min_carries:
        pos = random.randint(0, max_potential_carries - 1)
        if a_digits[pos] + b_digits[pos] < 10:  # not already set to cause a carry
            # set digits to force a carry at this position
            a_digits[pos] = random.randint(5, 9)
            b_digits[pos] = random.randint(10 - a_digits[pos], 9)
            carries_placed += 1

    # fill in remaining positions
    for i in range(max(digits_a, digits_b)):
        if a_digits[i] == 0 and b_digits[i] == 0:  # not already set for carries
            if i < digits_a and i < digits_b:
                # both numbers have this position
                # randomly decide if we want another carry
                if random.random() < 0.3 and carries_placed < max_potential_carries:
                    # add another carry
                    a_digits[i] = random.randint(5, 9)
                    b_digits[i] = random.randint(10 - a_digits[i], 9)
                    carries_placed += 1
                else:
                    # no carry
                    a_digits[i] = random.randint(0, 9)
                    b_digits[i] = random.randint(0, 9 - a_digits[i])
            elif i < digits_a:
                a_digits[i] = random.randint(1 if i == digits_a - 1 else 0, 9)
            elif i < digits_b:
                b_digits[i] = random.randint(1 if i == digits_b - 1 else 0, 9)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits[:digits_a])
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result
    result = a + b

    return a, b, result


def generate_no_borrow_negative_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """generate subtraction problems with negative numbers (-a - b) where no borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    # generate individual digits ensuring no borrowing
    a_digits = []
    b_digits = []

    for i in range(max(digits_a, digits_b)):
        if i < digits_a:
            if i < digits_b:
                # for positions where both numbers have digits
                a_digit = random.randint(0, 9)
                b_digit = random.randint(0, a_digit)
            else:
                # a has digit but b doesn't
                a_digit = random.randint(0, 9)
                b_digit = 0
        else:
            # a doesn't have digit but b does
            a_digit = 0
            b_digit = 0  # must be 0 to prevent borrowing

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits[:digits_a])
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result (for -a - b)
    result = -(a + b)

    return -a, -b, result


def generate_borrow_negative_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_borrows=1):
    """generate subtraction problems with negative numbers (-a - b) where borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    borrows_placed = 0
    max_potential_borrows = min(digits_a, digits_b)

    if min_borrows > max_potential_borrows:
        # can't satisfy the minimum borrows requirement
        digits_a = max(digits_a, min_borrows)
        digits_b = max(digits_b, min_borrows)
        max_potential_borrows = min(digits_a, digits_b)

    # generate individual digits ensuring at least one borrow
    a_digits = [0] * max(digits_a, digits_b)
    b_digits = [0] * max(digits_a, digits_b)

    # first, place required borrows
    while borrows_placed < min_borrows:
        pos = random.randint(0, max_potential_borrows - 1)
        if a_digits[pos] >= b_digits[pos]:  # not already set to cause a borrow
            # set digits to force a borrow at this position
            a_digits[pos] = random.randint(0, 4)
            b_digits[pos] = random.randint(a_digits[pos] + 1, 9)
            borrows_placed += 1

    # fill in remaining positions
    for i in range(max(digits_a, digits_b)):
        if a_digits[i] == 0 and b_digits[i] == 0:  # not already set for borrows
            if i < digits_a and i < digits_b:
                # both numbers have this position
                # randomly decide if we want another borrow
                if random.random() < 0.3 and borrows_placed < max_potential_borrows:
                    # add another borrow
                    a_digits[i] = random.randint(0, 4)
                    b_digits[i] = random.randint(a_digits[i] + 1, 9)
                    borrows_placed += 1
                else:
                    # no borrow
                    a_digits[i] = random.randint(0, 9)
                    b_digits[i] = random.randint(0, a_digits[i])
            elif i < digits_a:
                a_digits[i] = random.randint(1 if i == digits_a - 1 else 0, 9)
            elif i < digits_b:
                b_digits[i] = random.randint(1 if i == digits_b - 1 else 0, 9)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits[:digits_a])
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result (for -a - b)
    result = -(a + b)

    return -a, -b, result


def generate_no_borrow_positive_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """generate subtraction problems with positive numbers (a - b) where no borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b,
                           min(max_digits_b, digits_a))  # b cannot have more digits than a for a - b > 0

    # generate individual digits ensuring no borrowing
    a_digits = []
    b_digits = []

    for i in range(digits_a):
        if i < digits_b:
            # for positions where both numbers have digits
            b_digit = random.randint(0, 9)
            a_digit = random.randint(b_digit, 9)
        else:
            # a has digit but b doesn't
            a_digit = random.randint(1 if i == digits_a - 1 else 0, 9)
            b_digit = 0

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits)
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result
    result = a - b

    return a, -b, result


def generate_borrow_positive_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_borrows=1):
    """generate subtraction problems with positive numbers (a - b) where borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b,
                           min(max_digits_b, digits_a))  # b cannot have more digits than a for a - b > 0

    borrows_placed = 0
    max_potential_borrows = digits_b

    if min_borrows > max_potential_borrows:
        # can't satisfy the minimum borrows requirement
        digits_b = max(digits_b, min_borrows)
        digits_a = max(digits_a, digits_b)
        max_potential_borrows = digits_b

    # generate individual digits ensuring at least one borrow
    a_digits = [0] * digits_a
    b_digits = [0] * digits_a

    # first, place required borrows
    while borrows_placed < min_borrows:
        pos = random.randint(0, min(digits_b, digits_a) - 1)
        if a_digits[pos] >= b_digits[pos]:  # not already set to cause a borrow
            # set digits to force a borrow at this position
            a_digits[pos] = random.randint(0, 8)
            b_digits[pos] = random.randint(a_digits[pos] + 1, 9)
            borrows_placed += 1

    # ensure a > b by setting most significant digit of a appropriately
    highest_digit_b = 0
    for i in range(digits_b - 1, -1, -1):
        if b_digits[i] > 0:
            highest_digit_b = i
            break

    for i in range(digits_a - 1, highest_digit_b, -1):
        if i == digits_a - 1:
            a_digits[i] = random.randint(1, 9)  # ensure most significant digit of a is non-zero
        else:
            a_digits[i] = random.randint(0, 9)

    # fill in remaining positions
    for i in range(min(digits_a, digits_b)):
        if (a_digits[i] == 0 and b_digits[i] == 0) or (a_digits[i] >= b_digits[i] and i < highest_digit_b):
            # not already set for borrows or needs to be fixed
            if random.random() < 0.3 and borrows_placed < max_potential_borrows:
                # add another borrow
                a_digits[i] = random.randint(0, 8)
                b_digits[i] = random.randint(a_digits[i] + 1, 9)
                borrows_placed += 1
            else:
                # no additional borrow
                b_digits[i] = random.randint(0, 9)
                a_digits[i] = random.randint(b_digits[i], 9)

    # convert digit lists to numbers
    a = convert_digits_to_number(a_digits)
    b = convert_digits_to_number(b_digits[:digits_b])

    # calculate result
    result = a - b

    return a, -b, result


def generate_no_carry_negative_positive_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """generate addition problems with negative and positive numbers (-a + b) where |a| > |b|, no carrying."""
    # for -a + b without carrying, we can reuse the positive subtraction without borrowing logic
    # since -a + b = b - a mathematically
    a, b, result = generate_no_borrow_positive_subtraction(
        min_digits_a=min_digits_b,
        max_digits_a=max_digits_b,
        min_digits_b=min_digits_a,
        max_digits_b=max_digits_a
    )

    return -a, -b, -result  # This ensures we return (-a, b, result) for -a + b


def generate_carry_negative_positive_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3,
                                           min_carries=1):
    """generate addition problems with negative and positive numbers (-a + b) with carrying."""
    # for -a + b with carrying, we can simply reuse the positive subtraction with borrowing logic
    # since -a + b = b - a mathematically
    a, b, result = generate_borrow_positive_subtraction(
        min_digits_a=min_digits_b,
        max_digits_a=max_digits_b,
        min_digits_b=min_digits_a,
        max_digits_b=max_digits_a,
        min_borrows=min_carries
    )

    return -a, -b, -result