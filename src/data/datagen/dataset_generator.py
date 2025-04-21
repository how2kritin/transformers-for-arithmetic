import argparse
import time
from src.data.datagen.utils.arithmetic_generator_utils import set_random_seed
from src.data.datagen.utils.problem_generators import (
    generate_no_carry_addition,
    generate_carry_addition,
    generate_no_borrow_negative_subtraction,
    generate_borrow_negative_subtraction,
    generate_no_borrow_positive_subtraction,
    generate_borrow_positive_subtraction,
    generate_no_carry_negative_positive_addition,
    generate_carry_negative_positive_addition
)
from src.data.datagen.utils.dataset_utils import generate_dataset


def main():
    """main function to parse arguments and generate dataset."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description='generate arithmetic dataset for transformer training')
    parser.add_argument('--num_samples', type=int, default=80000,
                        help='total number of examples (divided by 8 for each category)')
    parser.add_argument('--min_digits', type=int, default=1,
                        help='minimum number of digits for regular dataset')
    parser.add_argument('--max_digits', type=int, default=5,
                        help='maximum number of digits for regular dataset')
    parser.add_argument('--output_dir', type=str, default="./datasets",
                        help='directory to save the dataset files')
    parser.add_argument('--filename', type=str, default="dataset.csv",
                        help='filename for the regular dataset')
    parser.add_argument('--gen_min_digits', type=int, default=None,
                        help='minimum number of digits for generalization dataset')
    parser.add_argument('--gen_max_digits', type=int, default=None,
                        help='maximum number of digits for generalization dataset')
    parser.add_argument('--gen_filename', type=str, default=None,
                        help='filename for the generalization dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    args = parser.parse_args()

    # set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"using random seed: {args.seed}")

    # default generalization digits if not specified but generalization path is provided
    if args.gen_filename and args.gen_min_digits is None and args.gen_max_digits is None:
        args.gen_min_digits = args.max_digits + 1
        args.gen_max_digits = args.max_digits + 3

    # define the categories and their generator functions
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

    # generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        output_dir=args.output_dir,
        filename=args.filename,
        gen_min_digits=args.gen_min_digits,
        gen_max_digits=args.gen_max_digits,
        gen_filename=args.gen_filename,
        categories=categories
    )

    # print execution time
    elapsed_time = time.time() - start_time
    print(f"\ndataset generation completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()