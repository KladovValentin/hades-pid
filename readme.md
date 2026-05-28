# ML-based PID for HADES particle candidates

This package trains and applies a neural-network particle-identification model
for HADES `HParticleCand`-like track candidates. It can be used in two ways:

1. as a standalone Python workflow for preparing data, training the network,
   exporting ONNX files, and producing prediction tables;
2. from a HADES analysis through the `HNetworkPID.h` / `HNetworkPID.C` wrapper,
   which loads the exported ONNX model and evaluates candidates directly in C++.

The network returns probabilities for five particle classes:

| Network index | Particle |
| --- | --- |
| 0 | pi+ |
| 1 | pi- |
| 2 | K+ |
| 3 | K- |
| 4 | p |

The basic candidate variables are momentum, charge, theta, MDC dE/dx, and beta.
The Python preparation code can also add derived beta-difference features. The
same normalization constants written during training must be used when the model
is applied.

## 1. Python Code

Use the Python code when you want to build or test the model without running a
HADES analysis job. The workflow is based on ROOT input read with `uproot`,
intermediate parquet tables, PyTorch training, and ONNX export.

Main files:

- `dataHandling.py`: builds training and application tables, selects the five
  PID classes, keeps the configured input variables, and writes normalization
  constants.
- `models/model.py`: contains the neural-network definitions. The current
  training setup uses a split domain-adversarial network: `Encoder`,
  `Classifier`, and `Discriminator`.
- `networkTrainer.py`: trains the network from simulation and experimental
  parquet tables, saves PyTorch weights, and exports ONNX models.
- `predict.py`: applies trained weights to parquet tables and provides plotting
  and validation helpers.


Train and export the model with:

```bash
python3 networkTrainer.py
```

The training step writes files in `nndata/`, for example:

```text
nndata/encoder.pt
nndata/classifier.pt
nndata/discriminator.pt
nndata/encoder.onnx
nndata/classifier.onnx
nndata/meanValues.txt
nndata/stdValues.txt
```


The prediction table contains the five class probabilities. For the split model,
it also contains the latent features produced by the encoder.

## 2. Use at HADES

For HADES analyses, use the companion C++ wrapper `HNetworkPID.h` /
`HNetworkPID.C`. The wrapper converts an `HParticleCand` into the network input,
normalizes it, runs the ONNX model with ONNX Runtime, and returns either the full
probability vector or PID indices selected from that vector.

The wrapper is not part of the Python package itself, but it is the intended C++
interface for using the exported model in an analysis.

### Include the wrapper

If your analysis uses both the header and source files, include the header in
your `.h` file:

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

Add the ONNX Runtime library to the Makefile:

```make
LIBS+=/lustre/hades/user/vkladov/packages/KinFit/lib/libonnxruntime.so
```

Export `MYHADDIR` when launching the job:

```bash
export MYHADDIR=/lustre/hades/user/vkladov/packages/KinFit
```

If you use your own third-party-packages directory, copy the ONNX Runtime `include/` and
library files there and export that directory as `MYHADDIR`.

### Run PID in an analysis

Create one `HNetworkPID` object and reuse it for all calls. The main
methods are:

- `getPredictionProbability(input)`: returns the five class probabilities.
- `getPredictionFull(x)`: returns the best PID index, or `-1` if the candidate
  is rejected or the best class is not separated enough.
- `getPredictionLooseFull(x)`: returns all PID indices with probability above
  `0.35`, sorted from highest to lowest probability.

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

Helper methods are available if you need lower-level access:

```cpp
bool nnInputIsGood(HParticleCand* x);
vector<float> get_NN_Input_Pars(HParticleCand* x);
vector<float> getPredictionProbability(vector<float> inputTensorValues);
```

`nnInputIsGood` applies basic range checks before inference. Candidates outside
the accepted range return `-1` for the strict prediction and an empty vector for
the loose prediction.

The wrapper expects the ONNX model files and normalization text files at the
paths configured inside `HNetworkPID.C`. Keep those paths as they are unless you
move the exported model or normalization files.
