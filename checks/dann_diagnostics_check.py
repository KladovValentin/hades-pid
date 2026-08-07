import math
import unittest

import torch

from dann_diagnostics import (
    alpha_schedule,
    balanced_domain_loss,
    conditional_domain_loss,
)
from models.model import (
    DomainAffineCorrection, MomentumDomainCorrection, ReverseLayerF,
)


class DannDiagnosticsCheck(unittest.TestCase):
    def test_gradient_reversal_scales_only_backward(self):
        values = torch.tensor([1.0, 2.0], requires_grad=True)
        output = ReverseLayerF.apply(values, 2.5)
        self.assertTrue(torch.equal(output, values))
        output.sum().backward()
        self.assertTrue(torch.allclose(values.grad, torch.full_like(values, -2.5)))

    def test_balanced_random_domain_loss_is_log_two(self):
        sim_x = torch.tensor([[-1.0], [1.0], [1.0]])
        exp_x = torch.tensor([[-1.0], [-1.0], [1.0]])
        sim_logits = torch.zeros((3, 2), requires_grad=True)
        exp_logits = torch.zeros((3, 2), requires_grad=True)
        loss = balanced_domain_loss(sim_logits, exp_logits, sim_x, exp_x, 0)
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)
        loss.backward()
        self.assertIsNotNone(sim_logits.grad)
        self.assertIsNotNone(exp_logits.grad)

    def test_conditional_loss_tolerates_absent_invalid_charge_classes(self):
        sim_x = torch.tensor([[-1.0], [-1.0], [1.0]])
        exp_x = torch.tensor([[-1.0], [1.0], [1.0]])
        sim_class = torch.tensor([1, 3, 0])
        exp_class = torch.tensor([1, 0, 2])
        loss = conditional_domain_loss(
            torch.zeros((3, 2)), torch.zeros((3, 2)),
            sim_x, exp_x, 0, sim_class, exp_class,
        )
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)

    def test_alpha_schedule_has_expected_endpoints(self):
        self.assertEqual(alpha_schedule(0, 100, 3.0), 0.0)
        self.assertGreater(alpha_schedule(99, 100, 3.0), 2.99)

    def test_domain_corrections_initialize_as_identity(self):
        values = torch.randn(12, 5)
        momentum_correction = MomentumDomainCorrection(5, momentum_index=0)
        for correction in (
            DomainAffineCorrection(5),
            momentum_correction,
        ):
            self.assertTrue(torch.equal(correction(values, 0.0), values))
            self.assertTrue(torch.equal(correction(values, 1.0), values))
        self.assertEqual(len(momentum_correction.shift_networks), 5)
        self.assertEqual(len(momentum_correction.scale_networks), 5)
        self.assertEqual(momentum_correction.shift_networks[0][0].out_features, 8)
        self.assertEqual(momentum_correction.shift_networks[0][-1].out_features, 1)

    def test_affine_correction_only_changes_simulation(self):
        correction = DomainAffineCorrection(3)
        with torch.no_grad():
            correction.raw_shift.fill_(0.5)
            correction.raw_scale.fill_(0.5)
        values = torch.ones(4, 3)
        self.assertTrue(torch.equal(correction(values, 0.0), values))
        self.assertFalse(torch.equal(correction(values, 1.0), values))


if __name__ == "__main__":
    unittest.main()
