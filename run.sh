export TRAINING_DATA=input/train_folds.csv
export MODEL=$1
export FOLD=0


python -m src.train
