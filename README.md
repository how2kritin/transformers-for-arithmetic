# transformers-for-arithmetic

## To generate data:
```bash
python3 -m src.data.datagen.dataset_generator --gen_filename dataset_generalization.csv
```

## To train a transformer on this data:
```bash
python3 -m src.train_model --epochs=100
```

## To run inference:
```bash
python -m src.inference --checkpoint checkpoints/best_model.pt --csv_file datasets/dataset_generalization.csv
```