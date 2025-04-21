import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import argparse


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

    return a, -b, result


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

    # Calculate result
    result = a - b

    return a, -b, result


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
    result = -a + b

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
    return -a, -b, -result


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


def calculate_correct_result(a, b):
    """Calculate the correct arithmetic result for a and b."""
    # Calculate the actual result
    return a + b  # This works for all cases since b is already negative for a-b or a is negative for -a+b


def generate_dataset(num_samples=10000,
                     min_digits=1,
                     max_digits=5,
                     save_path="arithmetic_dataset.csv",
                     gen_min_digits=None,
                     gen_max_digits=None,
                     gen_save_path=None):
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

            # Always calculate the correct result using simple a+b arithmetic
            # (b is already negative for subtractions)
            correct_result = calculate_correct_result(a, b)

            # Check if the generator's result is incorrect
            if result != correct_result:
                print(f"Calculation error fixed: {a} and {b} should result in {correct_result}, not {result}")
                result = correct_result

            # Store the category for logging but it won't be in the final output
            data.append({'expression': expression, 'result': result, '_category': category_name})

    # Generate generalization dataset if parameters are provided
    gen_data = []
    if gen_min_digits is not None and gen_max_digits is not None:
        gen_samples_per_category = num_samples // 16  # Half the number per category for generalization set

        for category_name, generator_func in tqdm(categories, desc="Generating generalization test set"):
            for _ in tqdm(range(gen_samples_per_category), desc=f"Generating {category_name} (generalization)"):
                if "carry" in category_name:
                    a, b, result = generator_func(min_digits_a=gen_min_digits, max_digits_a=gen_max_digits,
                                                  min_digits_b=gen_min_digits, max_digits_b=gen_max_digits,
                                                  min_carries=1)
                elif "borrow" in category_name:
                    a, b, result = generator_func(min_digits_a=gen_min_digits, max_digits_a=gen_max_digits,
                                                  min_digits_b=gen_min_digits, max_digits_b=gen_max_digits,
                                                  min_borrows=1)
                else:
                    a, b, result = generator_func(min_digits_a=gen_min_digits, max_digits_a=gen_max_digits,
                                                  min_digits_b=gen_min_digits, max_digits_b=gen_max_digits)

                # Format the expression
                expression = format_expression(a, b)

                # Always calculate the correct result using simple a+b arithmetic
                # (b is already negative for subtractions)
                correct_result = calculate_correct_result(a, b)

                # Check if the generator's result is incorrect
                if result != correct_result:
                    print(f"Calculation error fixed: {a} and {b} should result in {correct_result}, not {result}")
                    result = correct_result

                # Store the category for logging but it won't be in the final output
                gen_data.append(
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

    # Standard split for regular data
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

    # Save generalization dataset if it exists
    if gen_data and gen_save_path:
        gen_df = pd.DataFrame(gen_data)
        gen_df = gen_df.sample(frac=1).reset_index(drop=True)
        gen_df[['expression', 'result']].to_csv(gen_save_path, index=False)
        print(f"\nGeneralization dataset saved to {gen_save_path}")
        print(f"Generalization examples: {len(gen_df)}")

    # Print statistics
    print("\nCategory distribution:")
    category_counts = df['_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count / len(df) * 100:.1f}%)")

    if gen_data:
        print("\nGeneralization category distribution:")
        gen_category_counts = pd.DataFrame(gen_data)['_category'].value_counts()
        for category, count in gen_category_counts.items():
            print(f"{category}: {count} examples ({count / len(gen_data) * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate arithmetic dataset for transformer training')
    parser.add_argument('--num_samples', type=int, default=80000,
                        help='Total number of examples (divided by 8 for each category)')
    parser.add_argument('--min_digits', type=int, default=1,
                        help='Minimum number of digits for regular dataset')
    parser.add_argument('--max_digits', type=int, default=5,
                        help='Maximum number of digits for regular dataset')
    parser.add_argument('--save_path', type=str, default="arithmetic_dataset.csv",
                        help='Path to save the regular dataset')
    parser.add_argument('--gen_min_digits', type=int, default=None,
                        help='Minimum number of digits for generalization dataset')
    parser.add_argument('--gen_max_digits', type=int, default=None,
                        help='Maximum number of digits for generalization dataset')
    parser.add_argument('--gen_save_path', type=str, default=None,
                        help='Path to save the generalization dataset')

    args = parser.parse_args()

    # Default generalization digits if not specified but generalization path is provided
    if args.gen_save_path and args.gen_min_digits is None and args.gen_max_digits is None:
        args.gen_min_digits = args.max_digits + 1
        args.gen_max_digits = args.max_digits + 3

    generate_dataset(
        num_samples=args.num_samples,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        save_path=args.save_path,
        gen_min_digits=args.gen_min_digits,
        gen_max_digits=args.gen_max_digits,
        gen_save_path=args.gen_save_path
    )


if __name__ == "__main__":
    main()