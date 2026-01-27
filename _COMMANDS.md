# Commands

## Build SineWaves oracle Minari dataset (from config)

```bash
python datasets/discrete/build_dataset/sine_waves.py --config datasets/discrete/cfg/sine_waves/default.yml
```

If the dataset already exists and you want to rebuild it:

```bash
python datasets/discrete/build_dataset/sine_waves.py --config datasets/discrete/cfg/sine_waves/default.yml --overwrite
```

## Train BC seq2seq GRU policy (logs to MLflow)

```bash
source .venv/bin/activate
python train_seq2seq_discrete.py --dataset_config datasets/discrete/cfg/sine_waves/default.yml
```

