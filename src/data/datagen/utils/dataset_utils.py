import os

import pandas as pd
from tqdm import tqdm

from src.data.datagen.utils.arithmetic_generator_utils import format_expression, calculate_correct_result


def generate_data_for_category(category_name, generator_func, samples_count, min_digits, max_digits):
    """generate data for a specific category."""
    data = []
    for _ in tqdm(range(samples_count), desc=f"Generating {category_name}"):
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

        # always calculate the correct result using simple a+b arithmetic
        # (b is already negative for subtractions)
        correct_result = calculate_correct_result(a, b)

        # check if the generator's result is incorrect
        if result != correct_result:
            print(f"calculation error fixed: {a} and {b} should result in {correct_result}, not {result}")
            result = correct_result

        # store the category for logging purposes, but it won't be in the final output
        data.append({'expression': expression, 'result': result, '_category': category_name})

    return data


def save_dataset(df, save_path, base_name=None):
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
    test_ratio = 0.1

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

    print(f"dataset saved to {save_path}")
    print(f"total examples: {len(df)}")
    print(f"train set: {len(train_df)} examples")
    print(f"validation set: {len(val_df)} examples")
    print(f"test set: {len(test_df)} examples")

    return train_df, val_df, test_df


def print_category_statistics(df, label="category"):
    """print statistics about category distribution."""
    print(f"\n{label} distribution:")
    category_counts = df['_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} examples ({count / len(df) * 100:.1f}%)")


def generate_dataset(num_samples=10000, min_digits=1, max_digits=5, output_dir="./data",
                     filename="arithmetic_dataset.csv", gen_min_digits=None, gen_max_digits=None, gen_filename=None,
                     categories=None):
    """generate a balanced dataset across all provided categories."""
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # create full paths
    save_path = os.path.join(output_dir, filename)
    gen_save_path = os.path.join(output_dir, gen_filename) if gen_filename else None

    if not categories:
        raise ValueError("categories must be provided")

    samples_per_category = num_samples // len(categories)
    data = []

    # training and validation data generation
    for category_name, generator_func in tqdm(categories, desc="generating dataset by category"):
        category_data = generate_data_for_category(category_name, generator_func, samples_per_category, min_digits,
            max_digits)
        data.extend(category_data)

    # convert to dataframe
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

        for category_name, generator_func in tqdm(categories, desc="generating generalization test set"):
            category_data = generate_data_for_category(f"{category_name} (generalization)", generator_func,
                gen_samples_per_category, gen_min_digits, gen_max_digits)
            gen_data.extend(category_data)

        # convert to dataframe and shuffle
        gen_df = pd.DataFrame(gen_data)
        gen_df = gen_df.sample(frac=1).reset_index(drop=True)

        # save generalization dataset
        gen_df[['expression', 'result']].to_csv(gen_save_path, index=False)
        print(f"\ngeneralization dataset saved to {gen_save_path}")
        print(f"generalization examples: {len(gen_df)}")

        # print statistics for generalization dataset
        print_category_statistics(gen_df, "generalization category")

    return df
