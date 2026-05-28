# ML-based PID for HADES particle candidates

This repository contains Python code for training and applying a neural-network
PID model, plus a small C++ wrapper that runs the exported ONNX model inside a
HADES analysis.

The model classifies one particle candidate into five classes:

| Network index | Particle |
| --- | --- |
| 0 | pi+ |
| 1 | pi- |
| 2 | K+ |
| 3 | K- |
| 4 | p |

The HADES wrapper currently builds the network input from momentum, charge,
theta, MDC dE/dx, and beta. The Python dataset preparation can also add derived
beta-difference features, depending on the settings in `dataHandling.py`. In
both cases, the input is normalized with the mean and standard-deviation values
used during training.

## 1. Python code only

The Python part is useful independently from HADES if you want to prepare
training tables, train the model, export it to ONNX, or run predictions on
parquet tables.

Main files:

- `dataHandling.py`: reads ROOT data with `uproot`, selects the PID classes,
  keeps the input variables, writes parquet datasets, and stores the
  normalization constants in `nndata/`.
- `models/model.py`: defines the neural-network architectures. The currently
  used setup is a split domain-adversarial model with an `Encoder`,
  `Classifier`, and `Discriminator`.
- `networkTrainer.py`: trains the model from simulation and experimental
  parquet tables, saves PyTorch weights, and exports ONNX files.
- `predict.py`: loads trained weights, applies the model to a parquet table,
  writes prediction tables, and contains plotting/validation helpers.

Typical workflow:

1. Prepare or provide input parquet files in `nndata/`.
2. Make sure the dataset name in the scripts matches the files you want to use:

```python
dataSetType = 'NewKIsUsed'
```

3. Train and export the model:

```bash
python networkTrainer.py
```

This expects files such as:

```text
nndata/simuNewKIsUsed.parquet
nndata/expuNewKIsUsed.parquet
```

and produces files such as:

```text
nndata/encoderNewKIsUsed.pt
nndata/classifierNewKIsUsed.pt
nndata/discriminatorNewKIsUsed.pt
nndata/encoderNewKIsUsed.onnx
nndata/classifierNewKIsUsed.onnx
```

4. Apply a trained model from Python with the helpers in `predict.py`. The file
   currently contains executable plotting calls at the bottom, so use it as a
   script or edit those bottom calls for the prediction you want, for example:

```python
predict('expuNewKIsUsed.parquet', 'predictedExpNewKIsUsed.parquet')
```

The prediction output contains the five class probabilities and, for the split
model, the latent features written by the encoder.

## 2. Use inside HADES

The HADES side is implemented by `HNetworkPID.h` and `HNetworkPID.C`. It takes
an `HParticleCand`, builds the normalized input vector expected by the exported
model, runs the ONNX model, and returns PID probabilities or PID decisions.

### Include the wrapper

If you use both the header and source files, include the header in your `.h`
file:

```cpp
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.h"
HNetworkPID* networkPID;
```

Then include the source and create the object in your `.C` or `.cc` file:

```cpp
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();
```

If you use only the source file:

```cpp
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
HNetworkPID* networkPID = new HNetworkPID();
```

### Link ONNX Runtime

In the Makefile, link the ONNX Runtime library:

```make
LIBS+=/lustre/hades/user/vkladov/packages/KinFit/lib/libonnxruntime.so
```

When launching the job, export `MYHADDIR`:

```bash
export MYHADDIR=/lustre/hades/user/vkladov/packages/KinFit
```

Alternatively, copy the ONNX Runtime `include/` and library files to your own
third-party directory and export that directory as `MYHADDIR`.

### Use it in an analysis

For each `HParticleCand`, the wrapper can return:

- `getPredictionProbability(input)`: the five PID probabilities.
- `getPredictionFull(x)`: the single best PID index, or `-1` if the input is
  rejected or the best class is not sufficiently separated from the others.
- `getPredictionLooseFull(x)`: all PID indices with probability above `0.35`,
  sorted from highest to lowest probability.

Example:

```cpp
for (HParticleCand* x : particles) {
    vector<int> loosePids = networkPID->getPredictionLooseFull(x);
    int strictPid = networkPID->getPredictionFull(x);

    for (int pid : loosePids) {
        if (pid == 3) {
            kaonsNIdent.push_back(x);
        }
    }

    if (strictPid == 3) {
        hMKN->Fill(x->getMass2() * 1e-6);
    }
}
```

Useful helper methods:

```cpp
bool nnInputIsGood(HParticleCand* x);
vector<float> get_NN_Input_Pars(HParticleCand* x);
vector<float> getPredictionProbability(vector<float> inputTensorValues);
```

`nnInputIsGood` applies basic sanity cuts before inference. The current wrapper
rejects candidates with unphysical or out-of-training-range values, for example
bad mass squared, momentum, MDC dE/dx, or beta. Rejected candidates return `-1`
for the strict prediction and an empty vector for the loose prediction.

The C++ wrapper currently expects the ONNX model files and normalization text
files at the paths hard-coded in `HNetworkPID.C`; update those paths only when
you know where the files are installed for your analysis.
