import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os


def generate_no_carry_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """Generate addition problems where no carrying is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    # Generate individual digits ensuring no carrying
    a_digits = []
    b_digits = []

    for i in range(max(digits_a, digits_b)):
        if i < digits_a:
            # For digit positions in a
            if i < digits_b:
                # Ensure sum with corresponding digit in b is less than 10
                a_digit = random.randint(0, 9)
                b_digit = random.randint(0, 9 - a_digit)
            else:
                # No corresponding digit in b, any digit is fine
                a_digit = random.randint(0, 9)
                b_digit = 0
        else:
            # Position not in a
            a_digit = 0
            b_digit = random.randint(0, 9)

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits[:digits_a]):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result
    result = a + b

    return a, b, result


def generate_carry_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_carries=1):
    """Generate addition problems with at least one carry operation."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    carries_placed = 0
    max_potential_carries = min(digits_a, digits_b)

    if min_carries > max_potential_carries:
        # Can't satisfy the minimum carries requirement
        # Adjust parameters to ensure we can have at least one carry
        digits_a = max(digits_a, min_carries)
        digits_b = max(digits_b, min_carries)
        max_potential_carries = min(digits_a, digits_b)

    # Generate individual digits ensuring at least one carry
    a_digits = [0] * max(digits_a, digits_b)
    b_digits = [0] * max(digits_a, digits_b)

    # First, place required carries
    while carries_placed < min_carries:
        pos = random.randint(0, max_potential_carries - 1)
        if a_digits[pos] + b_digits[pos] < 10:  # Not already set to cause a carry
            # Set digits to force a carry at this position
            a_digits[pos] = random.randint(5, 9)
            b_digits[pos] = random.randint(10 - a_digits[pos], 9)
            carries_placed += 1

    # Fill in remaining positions
    for i in range(max(digits_a, digits_b)):
        if a_digits[i] == 0 and b_digits[i] == 0:  # Not already set for carries
            if i < digits_a and i < digits_b:
                # Both numbers have this position
                # Randomly decide if we want another carry
                if random.random() < 0.3 and carries_placed < max_potential_carries:
                    # Add another carry
                    a_digits[i] = random.randint(5, 9)
                    b_digits[i] = random.randint(10 - a_digits[i], 9)
                    carries_placed += 1
                else:
                    # No carry
                    a_digits[i] = random.randint(0, 9)
                    b_digits[i] = random.randint(0, 9 - a_digits[i])
            elif i < digits_a:
                a_digits[i] = random.randint(1 if i == digits_a - 1 else 0, 9)
            elif i < digits_b:
                b_digits[i] = random.randint(1 if i == digits_b - 1 else 0, 9)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits[:digits_a]):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result
    result = a + b

    return a, b, result


def generate_no_borrow_negative_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """Generate subtraction problems with negative numbers (-a - b) where no borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    # Generate individual digits ensuring no borrowing
    a_digits = []
    b_digits = []

    for i in range(max(digits_a, digits_b)):
        if i < digits_a:
            if i < digits_b:
                # For positions where both numbers have digits
                a_digit = random.randint(0, 9)
                b_digit = random.randint(0, a_digit)
            else:
                # A has digit but B doesn't
                a_digit = random.randint(0, 9)
                b_digit = 0
        else:
            # A doesn't have digit but B does
            a_digit = 0
            b_digit = 0  # Must be 0 to prevent borrowing

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits[:digits_a]):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result (for -a - b)
    result = -(a + b)

    return -a, -b, result


def generate_borrow_negative_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_borrows=1):
    """Generate subtraction problems with negative numbers (-a - b) where borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, max_digits_b)

    borrows_placed = 0
    max_potential_borrows = min(digits_a, digits_b)

    if min_borrows > max_potential_borrows:
        # Can't satisfy the minimum borrows requirement
        digits_a = max(digits_a, min_borrows)
        digits_b = max(digits_b, min_borrows)
        max_potential_borrows = min(digits_a, digits_b)

    # Generate individual digits ensuring at least one borrow
    a_digits = [0] * max(digits_a, digits_b)
    b_digits = [0] * max(digits_a, digits_b)

    # First, place required borrows
    while borrows_placed < min_borrows:
        pos = random.randint(0, max_potential_borrows - 1)
        if a_digits[pos] >= b_digits[pos]:  # Not already set to cause a borrow
            # Set digits to force a borrow at this position
            a_digits[pos] = random.randint(0, 4)
            b_digits[pos] = random.randint(a_digits[pos] + 1, 9)
            borrows_placed += 1

    # Fill in remaining positions
    for i in range(max(digits_a, digits_b)):
        if a_digits[i] == 0 and b_digits[i] == 0:  # Not already set for borrows
            if i < digits_a and i < digits_b:
                # Both numbers have this position
                # Randomly decide if we want another borrow
                if random.random() < 0.3 and borrows_placed < max_potential_borrows:
                    # Add another borrow
                    a_digits[i] = random.randint(0, 4)
                    b_digits[i] = random.randint(a_digits[i] + 1, 9)
                    borrows_placed += 1
                else:
                    # No borrow
                    a_digits[i] = random.randint(0, 9)
                    b_digits[i] = random.randint(0, a_digits[i])
            elif i < digits_a:
                a_digits[i] = random.randint(1 if i == digits_a - 1 else 0, 9)
            elif i < digits_b:
                b_digits[i] = random.randint(1 if i == digits_b - 1 else 0, 9)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits[:digits_a]):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result (for -a - b)
    result = -(a + b)

    return -a, -b, result


def generate_no_borrow_positive_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """Generate subtraction problems with positive numbers (a - b) where no borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b,
                              min(max_digits_b, digits_a))  # b cannot have more digits than a for a - b > 0

    # Generate individual digits ensuring no borrowing
    a_digits = []
    b_digits = []

    for i in range(digits_a):
        if i < digits_b:
            # For positions where both numbers have digits
            b_digit = random.randint(0, 9)
            a_digit = random.randint(b_digit, 9)
        else:
            # A has digit but B doesn't
            a_digit = random.randint(1 if i == digits_a - 1 else 0, 9)
            b_digit = 0

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result
    result = a - b

    return a, b, result


def generate_borrow_positive_subtraction(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3, min_borrows=1):
    """Generate subtraction problems with positive numbers (a - b) where borrowing is required."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b,
                              min(max_digits_b, digits_a))  # b cannot have more digits than a for a - b > 0

    borrows_placed = 0
    max_potential_borrows = digits_b

    if min_borrows > max_potential_borrows:
        # Can't satisfy the minimum borrows requirement
        digits_b = max(digits_b, min_borrows)
        digits_a = max(digits_a, digits_b)
        max_potential_borrows = digits_b

    # Generate individual digits ensuring at least one borrow
    a_digits = [0] * digits_a
    b_digits = [0] * digits_a

    # First, place required borrows
    while borrows_placed < min_borrows:
        pos = random.randint(0, min(digits_b, digits_a) - 1)
        if a_digits[pos] >= b_digits[pos]:  # Not already set to cause a borrow
            # Set digits to force a borrow at this position
            a_digits[pos] = random.randint(0, 8)
            b_digits[pos] = random.randint(a_digits[pos] + 1, 9)
            borrows_placed += 1

    # Ensure a > b by setting most significant digit of a appropriately
    highest_digit_b = 0
    for i in range(digits_b - 1, -1, -1):
        if b_digits[i] > 0:
            highest_digit_b = i
            break

    for i in range(digits_a - 1, highest_digit_b, -1):
        if i == digits_a - 1:
            a_digits[i] = random.randint(1, 9)  # Ensure most significant digit of a is non-zero
        else:
            a_digits[i] = random.randint(0, 9)

    # Fill in remaining positions
    for i in range(min(digits_a, digits_b)):
        if (a_digits[i] == 0 and b_digits[i] == 0) or (a_digits[i] >= b_digits[i] and i < highest_digit_b):
            # Not already set for borrows or needs to be fixed
            if random.random() < 0.3 and borrows_placed < max_potential_borrows:
                # Add another borrow
                a_digits[i] = random.randint(0, 8)
                b_digits[i] = random.randint(a_digits[i] + 1, 9)
                borrows_placed += 1
            else:
                # No additional borrow
                b_digits[i] = random.randint(0, 9)
                a_digits[i] = random.randint(b_digits[i], 9)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Ensure the result is correct and positive
    if a <= b:
        # If our algorithm failed to ensure a > b, just swap them
        return b, a, b - a

    # Calculate result
    result = a - b

    return a, b, result


def generate_no_carry_negative_positive_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3):
    """Generate addition problems with negative and positive numbers (-a + b) where |a| > |b|, no carrying."""
    digits_a = random.randint(min_digits_a, max_digits_a)
    digits_b = random.randint(min_digits_b, min(max_digits_b, digits_a))  # |b| cannot be greater than |a|

    # Generate individual digits ensuring no carrying
    a_digits = []
    b_digits = []

    for i in range(digits_a):
        if i < digits_b:
            # For positions where both numbers have digits
            b_digit = random.randint(0, 9)
            a_digit = random.randint(b_digit, 9)
        else:
            # A has digit but B doesn't
            a_digit = random.randint(1 if i == digits_a - 1 else 0, 9)
            b_digit = 0

        a_digits.append(a_digit)
        b_digits.append(b_digit)

    # Convert digit lists to numbers
    a = 0
    for digit in reversed(a_digits):
        a = a * 10 + digit

    b = 0
    for digit in reversed(b_digits[:digits_b]):
        b = b * 10 + digit

    # Calculate result (for -a + b)
    result = b - a

    return -a, b, result


def generate_carry_negative_positive_addition(min_digits_a=1, max_digits_a=3, min_digits_b=1, max_digits_b=3,
                                              min_carries=1):
    """Generate addition problems with negative and positive numbers (-a + b) with carrying."""
    # For -a + b with carrying, we can simply reuse the positive subtraction with borrowing logic
    # since -a + b = b - a mathematically
    a, b, result = generate_borrow_positive_subtraction(
        min_digits_a=min_digits_b,
        max_digits_a=max_digits_b,
        min_digits_b=min_digits_a,
        max_digits_b=max_digits_a,
        min_borrows=min_carries
    )

    # Swap a and b, and adjust signs
    return -b, a, result


def format_expression(a, b):
    """Format the expression string for the Transformer input."""
    if a >= 0 and b >= 0:
        # Both positive: a+b
        return f"{a}+{b}"
    elif a >= 0 and b < 0:
        # First positive, second negative: a-|b|
        return f"{a}-{abs(b)}"
    elif a < 0 and b >= 0:
        # First negative, second positive: -|a|+b
        return f"{a}+{b}"
    else:
        # Both negative: -|a|-|b|
        return f"{a}-{abs(b)}"


def fix_calculation(a, b, expected_result):
    """Verify that the calculation is correct."""
    if (a >= 0 and b >= 0) or (a < 0 and b < 0):
        # Both positive or both negative: addition
        actual_result = a + b
    elif a >= 0 and b < 0:
        # First positive, second negative: subtraction
        actual_result = a + b  # Since b is already negative
    else:  # a < 0 and b >= 0
        # First negative, second positive: addition
        actual_result = a + b

    if actual_result != expected_result:
        print(f"Calculation error: {a} and {b} should result in {actual_result}, not {expected_result}")
        return False
    return True


def generate_dataset(num_samples=10000, min_digits=1, max_digits=5, save_path="arithmetic_dataset.csv",
                     test_longer_digits=False):
    """Generate a balanced dataset across all 8 required cases."""
    samples_per_category = num_samples // 8

    data = []

    # Training and validation data generation
    categories = [
        ("a + b (no)", generate_no_carry_addition),
        ("a + b (carry)", generate_carry_addition),
        ("-a - b (no)", generate_no_borrow_negative_subtraction),
        ("-a - b (borrow)", generate_borrow_negative_subtraction),
        ("a - b (no)", generate_no_borrow_positive_subtraction),
        ("a - b (borrow)", generate_borrow_positive_subtraction),
        ("-a + b (no)", generate_no_carry_negative_positive_addition),
        ("-a + b (carry)", generate_carry_negative_positive_addition)
    ]

    for category_name, generator_func in tqdm(categories, desc="Generating dataset by category"):
        for _ in tqdm(range(samples_per_category), desc=f"Generating {category_name}"):
            if "carry" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits,
                                              min_digits_b=min_digits, max_digits_b=max_digits,
                                              min_carries=1)
            elif "borrow" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits,
                                              min_digits_b=min_digits, max_digits_b=max_digits,
                                              min_borrows=1)
            else:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits,
                                              min_digits_b=min_digits, max_digits_b=max_digits)

            # Format the expression
            expression = format_expression(a, b)

            # Verify the calculation is correct
            if not fix_calculation(a, b, result):
                # If there's an error, recalculate the correct result
                if (a >= 0 and b >= 0) or (a < 0 and b < 0):
                    result = a + b
                else:
                    result = a + b  # b is already negative for a-b or a is negative for -a+b

            # Store the category for logging but it won't be in the final output
            data.append({'expression': expression, 'result': result, '_category': category_name})

    # Generate a separate test set with longer digit numbers for generalization testing if requested
    if test_longer_digits:
        test_min_digits = max_digits + 1
        test_max_digits = max_digits + 3
        test_samples_per_category = num_samples // 16  # Half the number per category for test set

        for category_name, generator_func in tqdm(categories, desc="Generating generalization test set"):
            for _ in tqdm(range(test_samples_per_category), desc=f"Generating {category_name} (generalization)"):
                if "carry" in category_name:
                    a, b, result = generator_func(min_digits_a=test_min_digits, max_digits_a=test_max_digits,
                                                  min_digits_b=test_min_digits, max_digits_b=test_max_digits,
                                                  min_carries=1)
                elif "borrow" in category_name:
                    a, b, result = generator_func(min_digits_a=test_min_digits, max_digits_a=test_max_digits,
                                                  min_digits_b=test_min_digits, max_digits_b=test_max_digits,
                                                  min_borrows=1)
                else:
                    a, b, result = generator_func(min_digits_a=test_min_digits, max_digits_a=test_max_digits,
                                                  min_digits_b=test_min_digits, max_digits_b=test_max_digits)

                # Format the expression
                expression = format_expression(a, b)

                # Verify the calculation is correct
                if not fix_calculation(a, b, result):
                    # If there's an error, recalculate the correct result
                    if (a >= 0 and b >= 0) or (a < 0 and b < 0):
                        result = a + b
                    else:
                        result = a + b  # b is already negative for a-b or a is negative for -a+b

                # Store the category for logging but it won't be in the final output
                data.append(
                    {'expression': expression, 'result': result, '_category': category_name + " (generalization)"})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Create a copy without the category column for the output files
    output_df = df[['expression', 'result']].copy()

    # Save to CSV (without the category column)
    output_df.to_csv(save_path, index=False)

    print(f"Dataset saved to {save_path}")
    print(f"Total examples: {len(df)}")

    # Split into train, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    if test_longer_digits:
        # If we have a generalization test set, all the generalization examples go to test
        train_val_df = df[~df['_category'].str.contains("generalization")]
        gen_test_df = df[df['_category'].str.contains("generalization")]

        # Split the regular examples into train and validation
        train_size = int(len(train_val_df) * (train_ratio / (train_ratio + val_ratio)))
        train_df = train_val_df.iloc[:train_size]
        val_df = train_val_df.iloc[train_size:]
        test_df = gen_test_df
    else:
        # Standard split
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]

    # Save the splits (without the category column)
    base_name, ext = os.path.splitext(save_path)
    train_df[['expression', 'result']].to_csv(f"{base_name}_train{ext}", index=False)
    val_df[['expression', 'result']].to_csv(f"{base_name}_val{ext}", index=False)
    test_df[['expression', 'result']].to_csv(f"{base_name}_test{ext}", index=False)

    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")

    # Print statistics
    print("\nCategory distribution:")
    category_counts = df['_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    generate_dataset(
        num_samples=80000,  # Total number of examples (divided by 8 for each category)
        min_digits=1,  # Minimum number of digits
        max_digits=5,  # Maximum number of digits for training
        save_path="arithmetic_dataset.csv",
        test_longer_digits=True  # Generate additional test examples with more digits
    )