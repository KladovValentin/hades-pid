"""Fast, reproducible diagnostics for the Gen4 domain-adversarial model.

The production training script predates held-out domain validation.  This
module deliberately keeps experiments separate from deployment artifacts and
defaults to one epoch on ten percent of each parquet table.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from models.model import Classifier, Discriminator, Encoder


BASE_FEATURES = ["momentum", "charge", "theta", "mdcdedx", "beta"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_frame(path, fraction, seed, feature_set):
    frame = pd.read_parquet(path)
    if feature_set == "base":
        frame = frame[BASE_FEATURES + ["pid"]]
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")
    if fraction < 1.0:
        frame = frame.sample(frac=fraction, random_state=seed)
    return frame.reset_index(drop=True)


def split_and_normalize(sim, exp, seed, valid_fraction=0.2):
    features = list(sim.columns[:-1])
    if list(exp.columns[:-1]) != features:
        raise ValueError("simulation and experiment feature columns differ")

    sim_train = sim.sample(frac=1.0 - valid_fraction, random_state=seed)
    sim_valid = sim.drop(sim_train.index)
    exp_train = exp.sample(frac=1.0 - valid_fraction, random_state=seed)
    exp_valid = exp.drop(exp_train.index)

    # Fit one transform to the training portions of both domains.  Validation
    # information must not leak into the preprocessing constants.
    joined = pd.concat((sim_train[features], exp_train[features]), ignore_index=True)
    mean = joined.mean(axis=0)
    std = joined.std(axis=0, ddof=0).replace(0.0, 1.0)

    def convert(frame):
        x = ((frame[features] - mean) / std).to_numpy(dtype=np.float32)
        y = frame["pid"].to_numpy(dtype=np.int64)
        return x, y

    return (
        convert(sim_train), convert(exp_train),
        convert(sim_valid), convert(exp_valid),
        features, mean.to_dict(), std.to_dict(),
    )


def make_loader(data, batch_size, shuffle, seed):
    x, y = data
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def balanced_domain_loss(logits_sim, logits_exp, x_sim, x_exp, charge_index):
    """Average the four domain/charge groups with equal weight.

    The old loss concatenated simulation and experiment inside a charge group.
    Domain-dependent charge fractions then changed the implicit class prior.
    Equal group weighting makes log(2), rather than 2*log(2), the random-loss
    reference and prevents charge abundance from being a domain shortcut.
    """
    terms = []
    for logits, x, label in (
        (logits_sim, x_sim, 0),
        (logits_exp, x_exp, 1),
    ):
        for positive in (False, True):
            mask = (x[:, charge_index] > 0.0) == positive
            if mask.any():
                target = torch.full(
                    (int(mask.sum()),), label, dtype=torch.long, device=x.device
                )
                terms.append(nn.functional.cross_entropy(logits[mask], target))
    if len(terms) != 4:
        raise RuntimeError("a batch is missing a domain/charge group")
    return torch.stack(terms).mean()


def conditional_domain_loss(
    logits_sim, logits_exp, x_sim, x_exp, charge_index,
    sim_class, exp_class, number_of_classes=5,
):
    """Balance domains within charge and current predicted particle class.

    Gen4 simulation is a deliberately non-physical channel mixture, so its
    particle-class priors need not match inclusive data. A marginal adversary
    can otherwise identify the domain from particle identity and asks the
    encoder to erase precisely the information the classifier needs.
    """
    terms = []
    for class_index in range(number_of_classes):
        for positive in (False, True):
            sim_mask = (
                ((x_sim[:, charge_index] > 0.0) == positive)
                & (sim_class == class_index)
            )
            exp_mask = (
                ((x_exp[:, charge_index] > 0.0) == positive)
                & (exp_class == class_index)
            )
            # Only compare a conditional group represented in both domains.
            if not sim_mask.any() or not exp_mask.any():
                continue
            sim_target = torch.zeros(int(sim_mask.sum()), dtype=torch.long, device=x_sim.device)
            exp_target = torch.ones(int(exp_mask.sum()), dtype=torch.long, device=x_exp.device)
            terms.extend((
                nn.functional.cross_entropy(logits_sim[sim_mask], sim_target),
                nn.functional.cross_entropy(logits_exp[exp_mask], exp_target),
            ))
    if not terms:
        raise RuntimeError("a batch contains no shared pseudo-class/charge group")
    return torch.stack(terms).mean()


def make_domain_loss(
    conditioning, logits_sim, logits_exp, x_sim, x_exp, charge_index,
    sim_class=None, exp_class=None,
):
    if conditioning == "pseudo-class":
        return conditional_domain_loss(
            logits_sim, logits_exp, x_sim, x_exp, charge_index,
            sim_class, exp_class,
        )
    return balanced_domain_loss(logits_sim, logits_exp, x_sim, x_exp, charge_index)


def parameter_grad_norm(loss, parameters, retain_graph=False):
    params = [p for p in parameters if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    squared = sum(g.detach().pow(2).sum() for g in grads if g is not None)
    return float(torch.sqrt(squared).cpu())


def paired_batches(sim_loader, exp_loader):
    for sim_batch, exp_batch in zip(sim_loader, exp_loader):
        yield sim_batch, exp_batch


def alpha_schedule(step, total_steps, maximum):
    progress = step / max(total_steps - 1, 1)
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def train_epoch(
    mode, encoder, classifier, discriminator, sim_loader, exp_loader,
    main_optimizer, disc_optimizer, device, charge_index, domain_weight,
    alpha_max, disc_steps, conditioning,
):
    encoder.train()
    classifier.train()
    discriminator.train()
    totals = {"class_loss": 0.0, "domain_loss": 0.0, "steps": 0}
    total_steps = min(len(sim_loader), len(exp_loader))

    for step, ((sim_x, sim_y), (exp_x, _)) in enumerate(
        paired_batches(sim_loader, exp_loader)
    ):
        sim_x = sim_x.to(device)
        sim_y = sim_y.to(device)
        exp_x = exp_x.to(device)
        alpha = alpha_schedule(step, total_steps, alpha_max)

        # Detached pseudo-labels remove particle-mixture and charge-prior
        # shortcuts from the domain objective.
        sim_group = exp_group = None
        if conditioning == "pseudo-class":
            with torch.no_grad():
                sim_group = classifier(encoder(sim_x)).argmax(dim=1)
                exp_group = classifier(encoder(exp_x)).argmax(dim=1)

        if mode == "alternating":
            # Give the discriminator enough optimization capacity to remain a
            # meaningful adversary. Encoder features are detached here.
            with torch.no_grad():
                sim_detached = encoder(sim_x)
                exp_detached = encoder(exp_x)
            for _ in range(disc_steps):
                disc_optimizer.zero_grad(set_to_none=True)
                disc_loss = make_domain_loss(
                    conditioning,
                    discriminator.domain_logits(sim_detached),
                    discriminator.domain_logits(exp_detached),
                    sim_x, exp_x, charge_index, sim_group, exp_group,
                )
                disc_loss.backward()
                disc_optimizer.step()

        main_optimizer.zero_grad(set_to_none=True)
        sim_z = encoder(sim_x)
        exp_z = encoder(exp_x)
        class_loss = nn.functional.cross_entropy(classifier(sim_z), sim_y)

        if mode == "source-only":
            domain_loss = torch.zeros((), device=device)
            total_loss = class_loss
        else:
            if mode == "alternating":
                discriminator.requires_grad_(False)
            domain_loss = make_domain_loss(
                conditioning,
                discriminator(sim_z, alpha), discriminator(exp_z, alpha),
                sim_x, exp_x, charge_index, sim_group, exp_group,
            )
            total_loss = class_loss + domain_weight * domain_loss

        total_loss.backward()
        main_optimizer.step()

        if mode == "alternating":
            discriminator.requires_grad_(True)
        totals["class_loss"] += float(class_loss.detach().cpu())
        totals["domain_loss"] += float(domain_loss.detach().cpu())
        totals["steps"] += 1

    steps = max(totals.pop("steps"), 1)
    return {key: value / steps for key, value in totals.items()}


def warm_up_discriminator(
    encoder, classifier, discriminator, sim_loader, exp_loader,
    optimizer, device, charge_index, conditioning, epochs,
):
    """Fit the adversary against a frozen representation before GRL updates."""
    encoder.eval()
    classifier.eval()
    discriminator.train()
    losses = []
    for _ in range(epochs):
        for (sim_x, _), (exp_x, _) in paired_batches(sim_loader, exp_loader):
            sim_x, exp_x = sim_x.to(device), exp_x.to(device)
            with torch.no_grad():
                sim_z, exp_z = encoder(sim_x), encoder(exp_x)
                sim_group = exp_group = None
                if conditioning == "pseudo-class":
                    sim_group = classifier(sim_z).argmax(dim=1)
                    exp_group = classifier(exp_z).argmax(dim=1)
            optimizer.zero_grad(set_to_none=True)
            loss = make_domain_loss(
                conditioning,
                discriminator.domain_logits(sim_z),
                discriminator.domain_logits(exp_z),
                sim_x, exp_x, charge_index, sim_group, exp_group,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


@torch.no_grad()
def collect_outputs(encoder, classifier, discriminator, data, device, batch_size):
    encoder.eval()
    classifier.eval()
    discriminator.eval()
    loader = make_loader(data, batch_size, False, 0)
    latents, classes, domain_probabilities = [], [], []
    for x, _ in loader:
        x = x.to(device)
        z = encoder(x)
        latents.append(z.cpu())
        classes.append(classifier(z).argmax(dim=1).cpu())
        probabilities = discriminator.domain_logits(z).softmax(dim=1)[:, 1]
        domain_probabilities.append(probabilities.cpu())
    return (
        torch.cat(latents).numpy(),
        torch.cat(classes).numpy(),
        torch.cat(domain_probabilities).numpy(),
    )


class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LeakyReLU(),
            nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


def fit_probe(sim_values, exp_values, seed, device, max_per_domain, epochs):
    rng = np.random.default_rng(seed)

    def choose(values):
        count = min(len(values), max_per_domain)
        indices = rng.choice(len(values), size=count, replace=False)
        return values[indices]

    sim_values = choose(sim_values).astype(np.float32, copy=False)
    exp_values = choose(exp_values).astype(np.float32, copy=False)
    values = np.concatenate((sim_values, exp_values))
    labels = np.concatenate((np.zeros(len(sim_values)), np.ones(len(exp_values)))).astype(np.int64)
    order = rng.permutation(len(labels))
    split = len(order) // 2
    train_index, test_index = order[:split], order[split:]

    torch.manual_seed(seed)
    probe = Probe(values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    x_train = torch.from_numpy(values[train_index])
    y_train = torch.from_numpy(labels[train_index])
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=2048, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    probe.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(x), y)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(values[test_index]).to(device)
        scores = probe(test_x).softmax(dim=1)[:, 1].cpu().numpy()
    return probe, float(roc_auc_score(labels[test_index], scores)), values[test_index], labels[test_index]


@torch.no_grad()
def probe_scores(probe, values, device, batch_size=8192):
    scores = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size].astype(np.float32, copy=False)).to(device)
        scores.append(probe(x).softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def safe_auc(labels, scores, mask=None):
    if mask is not None:
        labels, scores = labels[mask], scores[mask]
    if len(labels) < 20 or np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def validation_metrics(
    encoder, classifier, discriminator, sim_valid, exp_valid, features,
    device, batch_size, seed, probe_samples, probe_epochs,
):
    sim_z, sim_prediction, sim_domain = collect_outputs(
        encoder, classifier, discriminator, sim_valid, device, batch_size
    )
    exp_z, exp_prediction, exp_domain = collect_outputs(
        encoder, classifier, discriminator, exp_valid, device, batch_size
    )
    domain_labels = np.concatenate((np.zeros(len(sim_z)), np.ones(len(exp_z))))
    domain_scores = np.concatenate((sim_domain, exp_domain))
    latent = np.concatenate((sim_z, exp_z))
    inputs = np.concatenate((sim_valid[0], exp_valid[0]))
    prediction = np.concatenate((sim_prediction, exp_prediction))
    charge_index = features.index("charge")
    charge = inputs[:, charge_index]

    latent_probe, latent_auc, _, _ = fit_probe(
        sim_z, exp_z, seed + 10, device, probe_samples, probe_epochs
    )
    latent_scores = probe_scores(latent_probe, latent, device)
    _, raw_auc, _, _ = fit_probe(
        sim_valid[0], exp_valid[0], seed + 20, device, probe_samples, probe_epochs
    )

    class_truth = sim_valid[1]
    class_accuracy = float(np.mean(sim_prediction == class_truth))
    per_class = {
        str(index): float(np.mean(sim_prediction[class_truth == index] == index))
        for index in np.unique(class_truth)
    }
    negative = charge < 0.0
    predicted_k_minus = prediction == 3

    negative_probe_auc = None
    if np.sum(charge[:len(sim_z)] < 0.0) >= 20 and np.sum(charge[len(sim_z):] < 0.0) >= 20:
        _, negative_probe_auc, _, _ = fit_probe(
            sim_z[charge[:len(sim_z)] < 0.0],
            exp_z[charge[len(sim_z):] < 0.0],
            seed + 30, device, probe_samples, probe_epochs,
        )
    k_minus_probe_auc = None
    sim_k_mask = sim_prediction == 3
    exp_k_mask = exp_prediction == 3
    if np.sum(sim_k_mask) >= 20 and np.sum(exp_k_mask) >= 20:
        _, k_minus_probe_auc, _, _ = fit_probe(
            sim_z[sim_k_mask], exp_z[exp_k_mask], seed + 40,
            device, probe_samples, probe_epochs,
        )
    return {
        "classification_accuracy": class_accuracy,
        "classification_accuracy_per_class": per_class,
        "discriminator_auc": safe_auc(domain_labels, domain_scores),
        "discriminator_probability_mean_sim": float(np.mean(sim_domain)),
        "discriminator_probability_mean_exp": float(np.mean(exp_domain)),
        "discriminator_probability_std_all": float(np.std(domain_scores)),
        "raw_input_probe_auc": raw_auc,
        "latent_probe_auc": latent_auc,
        "latent_probe_auc_negative_charge": safe_auc(domain_labels, latent_scores, negative),
        "latent_probe_auc_predicted_k_minus": safe_auc(
            domain_labels, latent_scores, predicted_k_minus
        ),
        "fresh_probe_auc_negative_charge": negative_probe_auc,
        "fresh_probe_auc_predicted_k_minus": k_minus_probe_auc,
        "predicted_k_minus_count_sim": int(np.sum(sim_prediction == 3)),
        "predicted_k_minus_count_exp": int(np.sum(exp_prediction == 3)),
    }


def gradient_metrics(
    encoder, classifier, discriminator, sim_loader, exp_loader,
    charge_index, device, conditioning, alpha_max, domain_weight,
):
    encoder.eval()
    classifier.eval()
    discriminator.eval()
    (sim_x, sim_y), (exp_x, _) = next(paired_batches(sim_loader, exp_loader))
    sim_x = sim_x.to(device)
    sim_y = sim_y.long().flatten().to(device)
    exp_x = exp_x.to(device)
    sim_z, exp_z = encoder(sim_x), encoder(exp_x)
    sim_logits = classifier(sim_z)
    exp_logits = classifier(exp_z)
    class_loss = nn.functional.cross_entropy(sim_logits, sim_y)
    domain_loss = make_domain_loss(
        conditioning,
        discriminator(sim_z, 1.0), discriminator(exp_z, 1.0),
        sim_x, exp_x, charge_index,
        sim_logits.detach().argmax(dim=1), exp_logits.detach().argmax(dim=1),
    )
    class_norm = parameter_grad_norm(class_loss, encoder.parameters(), retain_graph=True)
    domain_norm = parameter_grad_norm(domain_loss, encoder.parameters())
    ratio = domain_norm / max(class_norm, 1e-12)
    return {
        "encoder_class_gradient_norm": class_norm,
        "encoder_domain_gradient_norm_alpha_1": domain_norm,
        "domain_to_class_gradient_ratio_alpha_1": ratio,
        "effective_domain_to_class_gradient_ratio_at_alpha_max": (
            ratio * alpha_max * domain_weight
        ),
        "random_domain_loss_reference": math.log(2.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="nndata/simugen4.parquet")
    parser.add_argument("--exp", default="nndata/expugen4.parquet")
    parser.add_argument("--mode", choices=("source-only", "grl", "alternating"), default="alternating")
    parser.add_argument("--conditioning", choices=("charge", "pseudo-class"), default="charge")
    parser.add_argument("--feature-set", choices=("stored", "base"), default="stored")
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--domain-weight", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--discriminator-steps", type=int, default=3)
    parser.add_argument("--discriminator-warmup-epochs", type=int, default=0)
    parser.add_argument("--load-encoder")
    parser.add_argument("--load-classifier")
    parser.add_argument("--load-discriminator")
    parser.add_argument("--probe-samples", type=int, default=30000)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("dann_runs/latest.json"))
    parser.add_argument("--save-model-prefix", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = load_frame(args.sim, args.sample_fraction, args.seed, args.feature_set)
    exp = load_frame(args.exp, args.sample_fraction, args.seed + 1, args.feature_set)
    (
        sim_train, exp_train, sim_valid, exp_valid,
        features, normalization_mean, normalization_std,
    ) = split_and_normalize(sim, exp, args.seed)
    del sim, exp

    sim_loader = make_loader(sim_train, args.batch_size, True, args.seed)
    exp_loader = make_loader(exp_train, args.batch_size, True, args.seed + 1)
    sim_valid_loader = make_loader(sim_valid, args.batch_size, False, args.seed)
    exp_valid_loader = make_loader(exp_valid, args.batch_size, False, args.seed)
    charge_index = features.index("charge")

    encoder = Encoder(len(features), 64).to(device)
    classifier = Classifier(64, 5).to(device)
    discriminator = Discriminator(64, 2).to(device)
    for module, checkpoint in (
        (encoder, args.load_encoder),
        (classifier, args.load_classifier),
        (discriminator, args.load_discriminator),
    ):
        if checkpoint:
            module.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if args.mode == "alternating":
        main_parameters = list(encoder.parameters()) + list(classifier.parameters())
    else:
        main_parameters = (
            list(encoder.parameters()) + list(classifier.parameters())
            + list(discriminator.parameters())
        )
    main_optimizer = torch.optim.AdamW(
        main_parameters, lr=args.learning_rate, betas=(0.5, 0.99),
        weight_decay=args.weight_decay,
    )
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.discriminator_learning_rate,
        betas=(0.5, 0.99), weight_decay=args.weight_decay,
    )

    warmup_loss = warm_up_discriminator(
        encoder, classifier, discriminator, sim_loader, exp_loader,
        disc_optimizer, device, charge_index, args.conditioning,
        args.discriminator_warmup_epochs,
    )
    history = []
    for _ in range(args.epochs):
        history.append(train_epoch(
            args.mode, encoder, classifier, discriminator,
            sim_loader, exp_loader, main_optimizer, disc_optimizer,
            device, charge_index, args.domain_weight, args.alpha_max,
            args.discriminator_steps, args.conditioning,
        ))

    metrics = validation_metrics(
        encoder, classifier, discriminator, sim_valid, exp_valid, features,
        device, args.batch_size, args.seed, args.probe_samples, args.probe_epochs,
    )
    metrics.update(gradient_metrics(
        encoder, classifier, discriminator, sim_valid_loader, exp_valid_loader,
        charge_index, device, args.conditioning, args.alpha_max,
        args.domain_weight,
    ))
    result = {
        "config": {
            **vars(args),
            "output": str(args.output),
            "save_model_prefix": (
                str(args.save_model_prefix) if args.save_model_prefix else None
            ),
            "device": str(device),
        },
        "dataset": {
            "features": features,
            "simulation_train": len(sim_train[0]),
            "experiment_train": len(exp_train[0]),
            "simulation_validation": len(sim_valid[0]),
            "experiment_validation": len(exp_valid[0]),
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
        },
        "training": history,
        "discriminator_warmup_loss": warmup_loss,
        "validation": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_model_prefix:
        args.save_model_prefix.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), f"{args.save_model_prefix}_encoder.pt")
        torch.save(classifier.state_dict(), f"{args.save_model_prefix}_classifier.pt")
        torch.save(discriminator.state_dict(), f"{args.save_model_prefix}_discriminator.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
