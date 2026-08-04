# Gen4 domain-adversarial diagnostics

This branch adds a reproducible diagnostic harness in `dann_diagnostics.py`.
Its defaults are one epoch, 10% of each Gen4 parquet table, a fixed seed, and
an 80/20 train/validation split. Generated JSON and checkpoints are kept under
the ignored `dann_runs/` and `nndata/` directories.

## What was wrong

The stored domain discriminator being close to 0.5 is not evidence of domain
invariance. A fresh nonlinear probe still separates the frozen latent vectors.
The active production loop also does not use its validation loaders and always
overwrites the checkpoint at the final epoch.

The experiments found three additional problems:

1. A simultaneous optimizer lets the discriminator become an uninformative
   adversary. In a fresh one-epoch run its encoder domain-gradient norm was only
   0.15% of the classification-gradient norm.
2. The previous charge-balanced loss concatenated unequal simulation and data
   subsets inside each charge. The resulting domain prior was still affected by
   domain-dependent charge abundance. The diagnostic loss instead averages the
   four domain/charge groups equally; its random reference is `log(2)=0.6931`.
3. Gen4 simulation starts from a channel mixture that does not reproduce
   physical channel cross sections, but `dataHandling.py` then resamples the
   five simulated classes to explicit experimental target counts. The final
   simulation fractions are therefore intended to approximate experiment; the
   unlabeled experimental parquet does not allow that match to be verified
   directly. Residual class-prior or class-conditional kinematic differences
   remain possible, so global AUC must be accompanied by charge- and
   particle-candidate-group metrics rather than interpreted alone.

There is also a reproducibility mismatch: the note describes five inputs, but
`simugen4.parquet`, `expugen4.parquet`, and `encodergen4.pt` use eight. The extra
inputs are `newColPi`, `newColK`, and `newColp`, deterministic beta residuals.
A five-feature fresh model did not converge its rare classes adequately in the
one-epoch budget (88.4% overall accuracy and 0.6% K+ recall), so removing these
features is not supported by this quick study.

## Tested results

All values below use 10% of each table and one training epoch. Probe AUCs come
from fresh nonlinear networks that do not update the encoder.

| Configuration | Sim accuracy | Global latent AUC | Fresh negative-charge AUC | Fresh predicted-K- AUC |
|---|---:|---:|---:|---:|
| Stored checkpoint, seed 42 | 98.76% | 0.755 | 0.742 | 0.787 |
| Warm discriminator + 1 detached step | 98.76% | 0.744 | 0.710 | 0.765 |
| Warm discriminator + 2 detached steps | 98.75% | 0.751 | 0.704 | 0.721 |
| Warm discriminator + 3 detached steps | 98.76% | 0.740 | 0.663 | 0.732 |

The three-step configuration was repeated with seed 7. It retained 98.79%
simulation accuracy while changing global, negative-charge, and predicted-K-
AUC from 0.750/0.760/0.779 in the stored model to 0.743/0.686/0.634. The effect
therefore has the correct direction in both tested splits, though its magnitude
is seed- and subgroup-dependent.

The recommended diagnostic checkpoint is not yet a production model. On seed
42 its discriminator has AUC 0.576 and probability standard deviation 0.0445,
so it remains sensitive and has not reached 0.5 through constant output. Its
held-out probes also remain above chance, meaning one reduced-data epoch improves
but does not complete alignment.

## Recommended experiment

Run the selected warm-started configuration with:

```bash
python3 dann_diagnostics.py \
  --mode alternating \
  --conditioning charge \
  --sample-fraction 0.1 \
  --epochs 1 \
  --batch-size 4096 \
  --learning-rate 0.00003 \
  --discriminator-learning-rate 0.0005 \
  --discriminator-warmup-epochs 1 \
  --discriminator-steps 3 \
  --load-encoder nndata/encodergen4.pt \
  --load-classifier nndata/classifiergen4.pt \
  --save-model-prefix nndata/dann_diagnostic_recommended \
  --output dann_runs/recommended_disc3.json
```

This fits a fresh discriminator against a frozen stored representation, then
alternates three discriminator updates on detached features with one
encoder/classifier update. The discriminator is frozen during that latter
update, so gradient reversal changes the encoder without accidentally applying
the reversed objective to discriminator parameters.

Do not add the experiment/simulation label to the encoder input. It would give
the encoder a perfect domain shortcut and produce domain-specific, rather than
domain-invariant, features. The domain bit belongs only in the discriminator
target. If class-prior effects remain dominant, the next study should use tagged
experimental pions/kaons or validated pseudo-label conditioning and select a
checkpoint using subgroup probe AUC plus simulation class accuracy.
