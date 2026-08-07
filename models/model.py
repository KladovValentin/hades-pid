import torch.nn as nn
import torch
from torch.autograd import Function


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.nn_model = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, text):
        output = self.nn_model(text)
        return output



class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class DANN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DANN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(128, 128)
            #nn.Dropout(0.5),
            #nn.LeakyReLU(inplace=True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            #nn.BatchNorm1d(128),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, output_dim)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            #nn.BatchNorm1d(128),
            #torch.nn.Sigmoid()
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
            #torch.nn.Sigmoid()
            nn.LogSoftmax(dim=0)
        )

    
    def grad_reverse(self, x):
        return GradReverse.apply(x)

    def forward(self, input_data):
        feature = self.feature(input_data)
        reverse_feature = self.grad_reverse(feature)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature).squeeze()

        #print(domain_output.shape)

        return class_output, domain_output


def _domain_gate(input_data, domain):
    if domain is None:
        return torch.zeros_like(input_data[:, :1])
    if not torch.is_tensor(domain):
        return torch.full_like(input_data[:, :1], float(domain))
    gate = domain.to(device=input_data.device, dtype=input_data.dtype)
    if gate.ndim == 0:
        gate = gate.expand(input_data.shape[0]).unsqueeze(1)
    elif gate.ndim == 1:
        gate = gate.unsqueeze(1)
    return gate.clamp(0.0, 1.0)


class DomainAffineCorrection(nn.Module):
    """Bounded per-feature affine correction, initialized to identity."""

    def __init__(self, input_dim, max_shift=0.5, max_scale=0.25):
        super().__init__()
        self.raw_shift = nn.Parameter(torch.zeros(input_dim))
        self.raw_scale = nn.Parameter(torch.zeros(input_dim))
        self.max_shift = max_shift
        self.max_scale = max_scale

    def forward(self, input_data, domain):
        shift = self.max_shift * torch.tanh(self.raw_shift)
        scale = self.max_scale * torch.tanh(self.raw_scale)
        correction = shift + scale * input_data
        return input_data + _domain_gate(input_data, domain) * correction


class MomentumDomainCorrection(nn.Module):
    """Per-feature bounded corrections conditioned on normalized momentum."""

    def __init__(self, input_dim, momentum_index, max_shift=0.5,
                 max_scale=0.25):
        super().__init__()
        self.momentum_index = momentum_index
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.shift_networks = nn.ModuleList()
        self.scale_networks = nn.ModuleList()
        for _ in range(input_dim):
            shift_network = nn.Sequential(
                nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)
            )
            scale_network = nn.Sequential(
                nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)
            )
            nn.init.zeros_(shift_network[-1].weight)
            nn.init.zeros_(shift_network[-1].bias)
            nn.init.zeros_(scale_network[-1].weight)
            nn.init.zeros_(scale_network[-1].bias)
            self.shift_networks.append(shift_network)
            self.scale_networks.append(scale_network)

    def forward(self, input_data, domain):
        momentum = input_data[:, self.momentum_index:self.momentum_index + 1]
        raw_shift = torch.cat(
            [network(momentum) for network in self.shift_networks], dim=1
        )
        raw_scale = torch.cat(
            [network(momentum) for network in self.scale_networks], dim=1
        )
        shift = self.max_shift * torch.tanh(raw_shift)
        scale = self.max_scale * torch.tanh(raw_scale)
        correction = shift + scale * input_data
        return input_data + _domain_gate(input_data, domain) * correction


def infer_encoder_correction_mode(state_dict):
    """Infer the optional correction architecture stored in a checkpoint."""
    keys = state_dict.keys()
    if any(
        key.startswith((
            "domain_correction.shift_networks",
            "domain_correction.momentum_network",
        ))
        for key in keys
    ):
        return "momentum"
    if "domain_correction.raw_shift" in state_dict:
        return "affine"
    return "none"


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim, correction_mode="none",
                 momentum_index=0):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.correction_mode = correction_mode
        if correction_mode == "affine":
            self.domain_correction = DomainAffineCorrection(input_dim)
        elif correction_mode == "momentum":
            self.domain_correction = MomentumDomainCorrection(
                input_dim, momentum_index
            )
        elif correction_mode != "none":
            raise ValueError(
                "correction_mode must be 'none', 'affine', or 'momentum'"
            )

        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            #nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, output_dim),
            #nn.BatchNorm1d(output_dim),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(128, 128)
            #nn.Dropout(0.5),
            #nn.LeakyReLU(inplace=True)
        )
        """
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64, bias=True),
            #nn.BatchNorm1d(256),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, output_dim),
            #nn.BatchNorm1d(output_dim),
            #nn.LeakyReLU(inplace=True),
            #nn.Linear(128, 128)
            #nn.Dropout(0.5),
            #nn.LeakyReLU(inplace=True)
        )
        """

    def forward(self, input_data, domain=None):
        if self.correction_mode != "none":
            input_data = self.domain_correction(input_data, domain)
        feature = self.feature(input_data)

        return feature
    

class Classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        
        self.class_classifier = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            nn.BatchNorm1d(128),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, output_dim)
        )
        """
        self.class_classifier = nn.Sequential(
            nn.Linear(input_dim, 16, bias=True),
            nn.BatchNorm1d(16),
            #nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, output_dim)
        )
        """

    def forward(self, input_data):
        class_output = self.class_classifier(input_data)

        return class_output
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class Discriminator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, 128, bias=True),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            #nn.BatchNorm1d(128),
            #torch.nn.Sigmoid()
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def domain_logits(self, input_data):
        """Return domain logits without gradient reversal.

        This path is useful when the discriminator is optimized on detached
        encoder features.  Keeping it explicit also prevents an accidental
        discriminator update with the reversed encoder gradient.
        """
        return self.domain_classifier(input_data)

    def forward(self, input_data, alpha):
        reversed_input = ReverseLayerF.apply(input_data, alpha)
        #x = self.domain_classifier(reversed_input)
        #x = self.domain_classifier(input_data)
        #return x
        return self.domain_logits(reversed_input)
