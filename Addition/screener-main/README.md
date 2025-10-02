# Screener

## Installation
Make sure that you have installed [torch](https://pytorch.org/) compatible with your CUDA version.

Then, use:
```bash
git clone https://github.com/mishgon/screener.git && cd screener && pip install -e .
```

If you encounter any issues like `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, try:
```bash
pip uninstall opencv-python && pip install opencv-python-headless
```

Please ignore pip warnings about package versions conflicts.

## Training example on MOOD Abdomen

Download the dataset:
```bash
python scripts/download_mood_abdomen.py
```

For preprocessed data management we use [cotomka](https://github.com/mishgon/cotomka) package.
Prepared datasets are saved under a root directory specified in `~/.config/cotomka/cotomka.yaml` file.
Create this file with the following content (replace `/path/to/cotomka` with your desired location):
```yaml
root_dir: /path/to/cotomka
```

Then, run the preprocessing script:
```bash
python scripts/prepare_mood_abdomen.py
```

For experiments management we use [hydra](https://hydra.cc/) package.
Experiments are saved under a root directory specified in `configs/paths/default.yaml`:
```yaml
exp_dir: /path/to/experiments
```
Specify the desired path by replacing `/path/to/experiments`.

To train the DenseVICReg descriptor model, run:
```bash
python scripts/train_dense_vicreg_on_mood_abdomen.py
```
All the hyperparameters can be found in the training config `configs/train_dense_vicreg_on_mood_abdomen.yaml`.
After the training is complete, the descriptor model is saved to the file `/path/to/experiments/dense_vicreg_on_mood_abdomen/runs/<YYYY-MM-DD_HH-MM-SS>/descriptor_model.pt`, where `<YYYY-MM-DD_HH-MM-SS>` is date-time of the run.

To train the unconditional Glow density model, first, specify the path to the pretrained descriptor model in the training config `configs/train_glow_on_mood_abdomen.yaml`:
```yaml
descriptor_model_path: /path/to/experiments/dense_vicreg_on_mood_abdomen/runs/<YYYY-MM-DD_HH-MM-SS>/descriptor_model.pt
```
Then, run:
```bash
python scripts/train_glow_on_mood_abdomen.py
```
