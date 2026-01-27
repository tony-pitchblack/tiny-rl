# Commands

## Build SineWaves oracle Minari dataset (from config)

```bash
python datasets/discrete/build_dataset/sine_waves.py --config datasets/discrete/cfg/sine_waves/default.yml
```

If the dataset already exists and you want to rebuild it:

```bash
python datasets/discrete/build_dataset/sine_waves.py --config datasets/discrete/cfg/sine_waves/default.yml --overwrite
```

