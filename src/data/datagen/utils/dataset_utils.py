import os

import pandas as pd
from tqdm import tqdm

from src.data.datagen.utils.arithmetic_generator_utils import format_expression, calculate_correct_result


def generate_data_for_category(category_name, generator_func, samples_count, min_digits, max_digits,
                               existing_expressions=None):
    """Generate data for a specific category avoiding duplicates."""
    data = []
    attempts = 0
    max_attempts = samples_count * 10  # Limit attempts to avoid infinite loops

    if existing_expressions is None:
        existing_expressions = set()

    with tqdm(total=samples_count, desc=f"Generating {category_name}") as pbar:
        while len(data) < samples_count and attempts < max_attempts:
            attempts += 1

            # Generate expression using the appropriate generator function
            if "carry" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits, min_carries=1)
            elif "borrow" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits, min_borrows=1)
            else:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits)

            # Format the expression
            expression = format_expression(a, b)

            # Always calculate the correct result using simple a+b arithmetic
            # (b is already negative for subtractions)
            correct_result = calculate_correct_result(a, b)

            # Check if the generator's result is incorrect
            if result != correct_result:
                print(f"Calculation error fixed: {a} and {b} should result in {correct_result}, not {result}")
                result = correct_result

            # Check if this expression is a duplicate
            if expression not in existing_expressions:
                existing_expressions.add(expression)
                # Store the category for logging purposes, but it won't be in the final output
                data.append({'expression': expression, 'result': result, '_category': category_name})
                pbar.update(1)

    if len(data) < samples_count:
        print(
            f"Warning: Could only generate {len(data)}/{samples_count} unique examples for {category_name} after {attempts} attempts")

    return data


def save_dataset(df, save_path, base_name=None):
    """Save the dataset to csv file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Create a copy without the category column for the output files
    output_df = df[['expression', 'result']].copy()

    # Save to csv (without the category column)
    output_df.to_csv(save_path, index=False)

    # If base_name is not provided, extract it from save_path
    if base_name is None:
        base_name, ext = os.path.splitext(save_path)
    else:
        ext = os.path.splitext(save_path)[1]

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
    train_df[['expression', 'result']].to_csv(f"{base_name}_train{ext}", index=False)
    val_df[['expression', 'result']].to_csv(f"{base_name}_val{ext}", index=False)
    test_df[['expression', 'result']].to_csv(f"{base_name}_test{ext}", index=False)

    print(f"Dataset saved to {save_path}")
    print(f"Total examples: {len(df)}")
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")

    return train_df, val_df, test_df


def print_category_statistics(df, label="Category"):
    """Print statistics about category distribution."""
    print(f"\n{label} distribution:")
    category_counts = df['_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count / len(df) * 100:.1f}%)")


def generate_dataset(num_samples=10000, min_digits=1, max_digits=5, output_dir="./data",
                     filename="arithmetic_dataset.csv", gen_min_digits=None, gen_max_digits=None, gen_filename=None,
                     categories=None):
    """Generate a balanced dataset across all provided categories with no duplicates."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create full paths
    save_path = os.path.join(output_dir, filename)
    gen_save_path = os.path.join(output_dir, gen_filename) if gen_filename else None

    if not categories:
        raise ValueError("Categories must be provided")

    samples_per_category = num_samples // len(categories)
    data = []

    # Keep track of all expressions to avoid duplicates
    all_expressions = set()

    # Training and validation data generation
    for category_name, generator_func in tqdm(categories, desc="Generating dataset by category"):
        category_data = generate_data_for_category(category_name, generator_func, samples_per_category, min_digits,
            max_digits, all_expressions)
        data.extend(category_data)

    # Convert to dataframe
    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Save regular dataset
    base_name = os.path.join(output_dir, os.path.splitext(filename)[0])
    train_df, val_df, test_df = save_dataset(df, save_path, base_name)

    # Print statistics
    print_category_statistics(df)

    # Generate generalization dataset if parameters are provided
    if gen_min_digits is not None and gen_max_digits is not None and gen_filename is not None:
        gen_data = []
        gen_samples_per_category = num_samples // (
                len(categories) * 2)  # Half the number per category for generalization set

        # Create a new set for generalization expressions to ensure no overlap with training set
        gen_expressions = set(all_expressions)  # Start with training expressions to avoid duplicates

        for category_name, generator_func in tqdm(categories, desc="Generating generalization test set"):
            category_data = generate_data_for_category(f"{category_name} (generalization)", generator_func,
                gen_samples_per_category, gen_min_digits, gen_max_digits, gen_expressions)
            gen_data.extend(category_data)

        # Convert to dataframe and shuffle
        gen_df = pd.DataFrame(gen_data)
        gen_df = gen_df.sample(frac=1).reset_index(drop=True)

        # Save generalization dataset
        gen_df[['expression', 'result']].to_csv(gen_save_path, index=False)
        print(f"\nGeneralization dataset saved to {gen_save_path}")
        print(f"Generalization examples: {len(gen_df)}")

        # Print statistics for generalization dataset
        print_category_statistics(gen_df, "Generalization category")

    return df
