import argparse
import os

from models.ArithmeticTrainingAndInference.ArithmeticTransformerInference import ArithmeticTransformerInference
from src.models.ArithmeticTrainingAndInference.BaseModelHandler import BaseModelHandler
from src.models.ArithmeticTrainingAndInference.utils.model_utils import add_common_model_args


def main() -> None:
    """run inference with a trained arithmetic transformer model."""
    parser = argparse.ArgumentParser(description="run inference with arithmetic transformer")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to the model checkpoint file")
    parser.add_argument("--input", type=str, required=False,
                        help="input text for inference (e.g., '2+3')")
    parser.add_argument("--csv_file", type=str, required=False,
                        help="csv file containing 'expression' and 'result' columns")
    parser.add_argument("--output_file", type=str, required=False,
                        help="file to save output predictions")

    # add common arguments
    parser = add_common_model_args(parser)
    # override default batch size for inference
    parser.set_defaults(batch_size=256)

    args = parser.parse_args()

    # check for valid input options
    if not args.input and not args.csv_file:
        parser.error("at least one of --input or --csv_file must be provided")

    # set device
    device = BaseModelHandler.create_device(args.no_cuda)
    print(f"using device: {device}")

    # load model for inference
    inference_model = ArithmeticTransformerInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_seq_length=args.max_length,
        device=device
    )

    # perform inference based on input type
    if args.csv_file:
        # use a specific csv file
        print(f"evaluating model on {args.csv_file}...")
        results = inference_model.evaluate_csv_file(
            csv_file=args.csv_file,
            max_len=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # save predictions if output file is provided
        if args.output_file:
            # create base filename without extension
            base_filename = os.path.splitext(args.output_file)[0]
            correct_filename = f"{base_filename}_correct.txt"
            incorrect_filename = f"{base_filename}_incorrect.txt"
            summary_filename = args.output_file

            # count of correct and incorrect predictions
            correct_count = 0
            incorrect_count = 0

            # write summary file with overall metrics
            with open(summary_filename, 'w') as f:
                f.write(f"accuracy: {results['accuracy']:.4f}\n")
                f.write(f"character accuracy: {results['char_accuracy']:.4f}\n")
                f.write(f"perplexity: {results['perplexity']:.4f}\n\n")
                f.write(f"total examples: {results['num_samples']}\n")

            # write correct and incorrect predictions to separate files
            with open(correct_filename, 'w') as correct_file, open(incorrect_filename, 'w') as incorrect_file:
                # write headers to both files
                for file in [correct_file, incorrect_file]:
                    file.write("example\tinput\ttarget\tprediction\n")
                    file.write("-" * 60 + "\n")

                # write individual examples
                for i, (inp, target, pred) in enumerate(zip(
                        results['inputs'],
                        results['targets'],
                        results['predictions'])):

                    is_correct = pred.strip() == target.strip()
                    example_text = f"example {i + 1}:\n"
                    example_text += f"  input: {inp}\n"
                    example_text += f"  target: {target}\n"
                    example_text += f"  prediction: {pred}\n\n"

                    if is_correct:
                        correct_file.write(example_text)
                        correct_count += 1
                    else:
                        incorrect_file.write(example_text)
                        incorrect_count += 1

            # update summary file with count information
            with open(summary_filename, 'a') as f:
                f.write(f"correct predictions: {correct_count}\n")
                f.write(f"incorrect predictions: {incorrect_count}\n\n")
                f.write(f"correct predictions saved to: {os.path.basename(correct_filename)}\n")
                f.write(f"incorrect predictions saved to: {os.path.basename(incorrect_filename)}\n")

            print(f"results summary saved to {summary_filename}")
            print(f"correct predictions ({correct_count}) saved to {correct_filename}")
            print(f"incorrect predictions ({incorrect_count}) saved to {incorrect_filename}")

    elif args.input:
        # single input
        prediction = inference_model.predict_single(args.input, max_len=args.max_length)
        print(f"input: {args.input}")
        print(f"prediction: {prediction}")


if __name__ == "__main__":
    main()
