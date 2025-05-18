import os
from typing import Callable

import pandas as pd
from hyperframe.frame import DataFrame
from tqdm import tqdm

from src.data.datagen.utils.arithmetic_generator_utils import format_expression, calculate_correct_result


def generate_data_for_category(category_name: str, generator_func: Callable, samples_count: int, min_digits: int,
                               max_digits: int,
                               existing_expressions: set = None):
    """generate data for a specific category avoiding duplicates."""
    data = []
    attempts = 0
    max_attempts = samples_count * 10  # limit attempts to avoid infinite loops

    if existing_expressions is None:
        existing_expressions = set()

    with tqdm(total=samples_count, desc=f"generating {category_name}") as pbar:
        while len(data) < samples_count and attempts < max_attempts:
            attempts += 1

            # generate expression using the appropriate generator function
            if "carry" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits, min_carries=1)
            elif "borrow" in category_name:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits, min_borrows=1)
            else:
                a, b, result = generator_func(min_digits_a=min_digits, max_digits_a=max_digits, min_digits_b=min_digits,
                                              max_digits_b=max_digits)

            # format the expression
            expression = format_expression(a, b)

            # calculate the correct result
            correct_result = calculate_correct_result(a, b)

            # check if the generator's result is incorrect (should never trigger this case, but just in case (pun intended))
            if result != correct_result:
                print(f"Calculation error fixed: {a} and {b} should result in {correct_result}, not {result}")
                result = correct_result

            # check if this expression is a duplicate
            if expression not in existing_expressions:
                existing_expressions.add(expression)
                # store the category for logging purposes, but it won't be in the final output
                data.append({'expression': expression, 'result': result, '_category': category_name})
                pbar.update(1)

    if len(data) < samples_count:
        print(
            f"Warning: Could only generate {len(data)}/{samples_count} unique examples for {category_name} after {attempts} attempts")

    return data


def save_dataset(df: DataFrame, save_path: str, base_name: str = None):
    """save the dataset to csv file."""
    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # create a copy without the category column for the output files
    output_df = df[['expression', 'result']].copy()

    # save to csv (without the category column)
    output_df.to_csv(save_path, index=False)

    # if base_name is not provided, extract it from save_path
    if base_name is None:
        base_name, ext = os.path.splitext(save_path)
    else:
        ext = os.path.splitext(save_path)[1]

    # split into train, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1

    # standard split for regular data
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # save the splits (without the category column)
    train_df[['expression', 'result']].to_csv(f"{base_name}_train{ext}", index=False)
    val_df[['expression', 'result']].to_csv(f"{base_name}_val{ext}", index=False)
    test_df[['expression', 'result']].to_csv(f"{base_name}_test{ext}", index=False)

    print(f"Dataset saved to {save_path}")
    print(f"Total examples: {len(df)}")
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")

    return train_df, val_df, test_df


def print_category_statistics(df: DataFrame, label: str = "Category"):
    """print statistics about category distribution"""
    print(f"\n{label} distribution:")
    category_counts = df['_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count / len(df) * 100:.1f}%)")


def generate_dataset(num_samples: int = 10000, min_digits: int = 1, max_digits: int = 5, output_dir: str = "./data",
                     filename: str = "arithmetic_dataset.csv", gen_min_digits: int = None, gen_max_digits: int = None,
                     gen_filename: str = None,
                     categories: list[str] = None):
    """generate a balanced dataset across all provided categories with no duplicates"""
    os.makedirs(output_dir, exist_ok=True)

    # create full paths
    save_path = os.path.join(output_dir, filename)
    gen_save_path = os.path.join(output_dir, gen_filename) if gen_filename else None

    if not categories:
        raise ValueError("Categories must be provided")

    samples_per_category = num_samples // len(categories)
    data = []

    # keep track of all expressions to avoid duplicates
    all_expressions = set()

    # training and validation data generation
    for category_name, generator_func in tqdm(categories, desc="Generating dataset by category"):
        category_data = generate_data_for_category(category_name, generator_func, samples_per_category, min_digits,
                                                   max_digits, all_expressions)
        data.extend(category_data)

    df = pd.DataFrame(data)

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # save regular dataset
    base_name = os.path.join(output_dir, os.path.splitext(filename)[0])
    _, _, _ = save_dataset(df, save_path, base_name)

    # print statistics
    print_category_statistics(df)

    # generate generalization dataset if parameters are provided
    if gen_min_digits is not None and gen_max_digits is not None and gen_filename is not None:
        gen_data = []
        gen_samples_per_category = num_samples // (
                len(categories) * 2)  # half the number per category for generalization set

        # create a new set for generalization expressions to ensure no overlap with training set
        gen_expressions = set(all_expressions)  # start with training expressions to avoid duplicates

        for category_name, generator_func in tqdm(categories, desc="Generating generalization test set"):
            category_data = generate_data_for_category(f"{category_name} (generalization)", generator_func,
                                                       gen_samples_per_category, gen_min_digits, gen_max_digits,
                                                       gen_expressions)
            gen_data.extend(category_data)

        # convert to dataframe and shuffle
        gen_df = pd.DataFrame(gen_data)
        gen_df = gen_df.sample(frac=1).reset_index(drop=True)

        # save generalization dataset
        gen_df[['expression', 'result']].to_csv(gen_save_path, index=False)
        print(f"\nGeneralization dataset saved to {gen_save_path}")
        print(f"Generalization examples: {len(gen_df)}")

        # print statistics for generalization dataset
        print_category_statistics(gen_df, "Generalization category")

    return df
