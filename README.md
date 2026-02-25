# Differentially Private Synthetic Data Generation (DPSDG)

![DPSDG overview](https://raw.githubusercontent.com/brains-group/dpsdg/main/dpctgan.png)

This repository contains the official implementation of the paper "Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation" as seen in TIME workshop of WebConf 2026. The code implements the same specs as the original CTGAN/TVAE model from the `ctgan` package.

## Installation

### From PyPI

```bash
pip install dpsdg
```

### From Source

Clone the repository and install it locally:

```bash
git clone https://github.com/brains-group/dpsdg.git
cd dpsdg
pip install .
```

Or install directly from GitHub without cloning first:

```bash
pip install git+https://github.com/brains-group/dpsdg.git
```

## Usage

This package extends [CTGAN](https://github.com/sdv-dev/CTGAN) with differentially private training via DP-SGD. The API closely mirrors the original, so the only change is swapping the model class and adding a few privacy parameters.

Your data should be a pandas `DataFrame` with:

- Continuous columns as `float`
- Discrete/categorical columns as `int` or `str`
- No missing values

### DP-CTGAN

**Original CTGAN** (no privacy):

```python
from ctgan import CTGAN

ctgan = CTGAN(epochs=300)
ctgan.fit(real_data, discrete_columns)
synthetic_data = ctgan.sample(1000)
```

**DP-CTGAN** (with differential privacy):

```python
from dpsdg.models.dp_ctgan import IterDPCTGAN

model = IterDPCTGAN(epsilon=1.0, delta=1e-5, epochs=300)
model.fit_transformer(real_data, discrete_columns)  # must be called before fit
model.fit(real_data, discrete_columns)
synthetic_data = model.sample(1000)
```

The key difference is the `fit_transformer` call before `fit`. This sets up the privacy-aware data transformer. After that, `fit` and `sample` work exactly as in the original.

### DP-TVAE

**Original TVAE** (no privacy):

```python
from ctgan import TVAE

tvae = TVAE(epochs=300)
tvae.fit(real_data, discrete_columns)
synthetic_data = tvae.sample(1000)
```

**DP-TVAE** (with differential privacy):

```python
from dpsdg.models.dp_tvae import IterTVAE

model = IterTVAE(epsilon=1.0, delta=1e-5, epochs=300)
model.fit(real_data, discrete_columns)
synthetic_data = model.sample(1000)
```

## Privacy Parameters

Both models add DP-specific parameters on top of the standard CTGAN/TVAE arguments.

### Shared Parameters

| Parameter       | Type    | Default                      | Description                                                                                                                                                                                                  |
| --------------- | ------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `epsilon`       | `float` | `0.0` (CTGAN) / `1.0` (TVAE) | Privacy budget (ε). Smaller values give a stronger privacy guarantee at the cost of data utility. Setting `epsilon=0.0` disables DP noise entirely. Common choices are `1.0`, `5.0`, or `10.0`.              |
| `delta`         | `float` | `1e-5`                       | Failure probability (δ) for the DP guarantee. Should be much smaller than `1/n`, where `n` is the number of training rows.                                                                                   |
| `max_grad_norm` | `float` | `1.0`                        | Per-sample gradient clipping threshold used in DP-SGD. This bounds the sensitivity of each update — larger values preserve more gradient signal but require proportionally more noise to achieve the same ε. |

### DP-CTGAN Only

| Parameter              | Type    | Default | Description                                                                                                                                                          |
| ---------------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_gradient_penalty` | `bool`  | `True`  | Enables a WGAN-GP style gradient penalty on the discriminator. Recommended to keep enabled, as it stabilizes GAN training under the high noise levels typical of DP. |
| `gp_lambda`            | `float` | `10`    | Weight for the gradient penalty term in the discriminator loss. Higher values enforce the Lipschitz constraint more strongly.                                        |

### DP-TVAE Only

| Parameter              | Type   | Default | Description                                                                                                                                                     |
| ---------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_opacus_noise_mul` | `bool` | `False` | When `True`, delegates noise multiplier calculation to Opacus. By default the package computes it directly via `opacus.accountants.utils.get_noise_multiplier`. |
