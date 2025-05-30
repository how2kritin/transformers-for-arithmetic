import argparse

from models.ArithmeticTrainingAndInference.ArithmeticTransformerTrainer import ArithmeticTransformerTrainer
from src.data.processing.dataloader import create_dataloaders
from src.data.processing.tokenizer import ArithmeticTokenizer
from src.models.ArithmeticTrainingAndInference.BaseModelHandler import BaseModelHandler
from src.models.ArithmeticTrainingAndInference.utils.model_utils import add_common_model_args, \
    add_model_architecture_args, print_model_info
from src.models.ArithmeticTransformer.ArithmeticTransformer import ArithmeticTransformer


def main() -> None:
    """main function to train the model."""
    parser = argparse.ArgumentParser(description="train arithmetic transformer")

    # data arguments
    parser.add_argument("--dataset_path", type=str, default="./datasets",
                        help="path to the dataset directory")

    # add common arguments using utility functions
    parser = add_model_architecture_args(parser)
    parser = add_common_model_args(parser)

    # training-specific arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--resume", type=str, help="resume from checkpoint file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="directory to save checkpoints")

    args = parser.parse_args()

    # set device using utility
    device = BaseModelHandler.create_device(args.no_cuda)
    print(f"using device: {device}")

    # create tokenizer
    tokenizer = ArithmeticTokenizer(max_length=args.max_seq_length)

    # create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=args.num_workers
    )

    # print dataset information
    print(f"training batches: {len(train_loader)}")
    if val_loader:
        print(f"validation batches: {len(val_loader)}")
    if test_loader:
        print(f"test batches: {len(test_loader)}")

    # initialize model
    model = ArithmeticTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding_type
    )

    # print model info using utility
    print_model_info(model, vars(args))

    # initialize trainer
    trainer = ArithmeticTransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # load checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        trainer.plot_training_history()

    # train model
    trainer.train(epochs=args.epochs, early_stopping_patience=args.patience)

    print("training completed!")


if __name__ == "__main__":
    main()
