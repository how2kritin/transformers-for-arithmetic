import pandas as pd
import argparse
from tqdm import tqdm


def perform_arithmetic(expression):
    """
    Perform the arithmetic operation expressed in the string.

    Args:
        expression (str): An arithmetic expression like "123+45" or "-9-7"

    Returns:
        int: The result of the arithmetic operation
    """
    # Handle edge cases
    if not expression:
        raise ValueError("Empty expression")

    # Parse the expression to find the operator and operands
    if '+' in expression:
        # Addition
        parts = expression.split('+')
        if len(parts) != 2:
            raise ValueError(f"Invalid addition expression: {expression}")

        a = int(parts[0])
        b = int(parts[1])
        return a + b
    elif '-' in expression:
        # Find the first occurrence of '-' (might be a negative number)
        first_minus_pos = expression.find('-')

        # Check if there's a second minus
        if first_minus_pos == 0:
            # First character is '-', look for second minus
            second_minus_pos = expression.find('-', 1)
            if second_minus_pos == -1:
                raise ValueError(f"Invalid subtraction expression: {expression}")

            a = int(expression[:second_minus_pos])
            b = int(expression[second_minus_pos + 1:])
            return a - b
        else:
            # Normal subtraction a-b
            parts = expression.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid subtraction expression: {expression}")

            a = int(parts[0])
            b = int(parts[1])
            return a - b
    else:
        raise ValueError(f"No operator found in expression: {expression}")


def check_dataset(file_path):
    """
    Check if the arithmetic operations in the dataset are correctly computed.

    Args:
        file_path (str): Path to the CSV file containing 'expression' and 'result' columns

    Returns:
        tuple: (number of correct examples, number of incorrect examples, list of errors)
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading the file {file_path}: {e}")
        return 0, 0, []

    # Check if the required columns exist
    if 'expression' not in df.columns or 'result' not in df.columns:
        print(f"Error: The file must contain 'expression' and 'result' columns")
        return 0, 0, []

    correct_count = 0
    incorrect_count = 0
    errors = []

    # Check each row
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Checking dataset"):
        expression = str(row['expression'])
        expected_result = int(row['result'])

        try:
            actual_result = perform_arithmetic(expression)

            if actual_result == expected_result:
                correct_count += 1
            else:
                incorrect_count += 1
                errors.append({
                    'index': i,
                    'expression': expression,
                    'expected': expected_result,
                    'actual': actual_result
                })
        except Exception as e:
            incorrect_count += 1
            errors.append({
                'index': i,
                'expression': expression,
                'expected': expected_result,
                'error': str(e)
            })

    return correct_count, incorrect_count, errors


def main():
    parser = argparse.ArgumentParser(description='Check arithmetic dataset for correctness')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--output', type=str, help='Path to save error report (optional)')
    parser.add_argument('--max-errors', type=int, default=10, help='Maximum number of errors to display (default: 10)')

    args = parser.parse_args()

    print(f"Checking file: {args.file_path}")
    correct_count, incorrect_count, errors = check_dataset(args.file_path)

    total = correct_count + incorrect_count
    accuracy = (correct_count / total) * 100 if total > 0 else 0

    print(f"\nResults:")
    print(f"Total examples: {total}")
    print(f"Correct examples: {correct_count} ({accuracy:.2f}%)")
    print(f"Incorrect examples: {incorrect_count} ({100 - accuracy:.2f}%)")

    if errors:
        print(f"\nFound {len(errors)} errors. Here are the first {min(args.max_errors, len(errors))}:")
        for i, error in enumerate(errors[:args.max_errors]):
            if 'error' in error:
                print(
                    f"{i + 1}. Index {error['index']}: Expression '{error['expression']}' - Expected {error['expected']} - Error: {error['error']}")
            else:
                print(
                    f"{i + 1}. Index {error['index']}: Expression '{error['expression']}' - Expected {error['expected']} - Got {error['actual']}")

    # Save error report if requested
    if args.output and errors:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(args.output, index=False)
        print(f"\nDetailed error report saved to {args.output}")


if __name__ == "__main__":
    main()