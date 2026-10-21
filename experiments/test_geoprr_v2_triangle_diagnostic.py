"""Two bounded CPU checks for the frozen-triangle conditional diagnostic."""
import json
import math
import unittest

import torch

from experiments.geoprr_v2_triangle_diagnostic import summarize_triangle


COUNTS = {"triangle_summary_calls": 0, "reference_blocks": 0, "summary_image_equivalents": 0}


def summarize(*args, **kwargs):
    result = summarize_triangle(*args, **kwargs)
    for key, value in result["counts"].items():
        COUNTS[key] += value
    return result


def enumerate_conditionals(log_a, log_p, log_b, log_factor, atoms, labels, masks):
    bins = log_a.shape[-1]
    weight = torch.stack([torch.stack([torch.stack([
        log_p[:, a, d] + log_b[:, a, s] + log_factor[:, (a + d) % bins, (a + s) % bins]
        for s in range(bins)], -1) for d in range(bins)], -2) for a in range(bins)], 1)
    conditional = weight.flatten(2).softmax(-1).reshape_as(weight)
    log_z_a = log_a + weight.flatten(2).logsumexp(-1)
    reference = log_z_a.softmax(-1)
    readings = (conditional * atoms[None, None]).sum((2, 3))
    invalid = (conditional * masks["invalid"][None, None]).sum((2, 3))
    outside = (conditional * masks["outside"][None, None]).sum((2, 3))
    correct_p, correct_b, correct_joint = [], [], []
    for a in range(bins):
        p_support = torch.stack([labels[:, 1, (a + d) % bins] > 0 for d in range(bins)], -1)
        b_support = torch.stack([labels[:, 2, (a + s) % bins] > 0 for s in range(bins)], -1)
        correct_p.append((conditional[:, a] * p_support[:, :, None]).sum((1, 2)))
        correct_b.append((conditional[:, a] * b_support[:, None, :]).sum((1, 2)))
        correct_joint.append((conditional[:, a] * p_support[:, :, None] * b_support[:, None, :]).sum((1, 2)))
    return {"A": reference, "conditional_reading": readings, "reading": (reference * readings).sum(-1),
            "GTA_reading": (labels[:, 0] * readings).sum(-1), "logZ_a": log_z_a,
            "conditional_P_correct_mass": torch.stack(correct_p, 1),
            "conditional_B_correct_mass": torch.stack(correct_b, 1),
            "joint_correct_mass": torch.stack(correct_joint, 1),
            "conditional_invalid_mass": invalid, "conditional_outside_arc_mass": outside,
            "invalid_mass": (reference * invalid).sum(-1), "outside_arc_mass": (reference * outside).sum(-1),
            "GTA_invalid_mass": (labels[:, 0] * invalid).sum(-1),
            "GTA_outside_arc_mass": (labels[:, 0] * outside).sum(-1)}


def reading_atoms(bins):
    delta = torch.arange(bins, dtype=torch.float64) * (2 * math.pi / bins)
    span = delta[None]
    raw = ((delta[:, None] - span / 2 + math.pi).remainder(2 * math.pi) - math.pi + span / 2) / span.masked_fill(span == 0, 1)
    atoms = raw.clamp(0, 1)
    atoms[:, 0] = .5
    invalid = (span == 0).expand(bins, -1)
    return atoms, {"invalid": invalid, "outside": ((raw < 0) | (raw > 1)) & ~invalid}


class TriangleDiagnosticTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"CPU_operator_tests": 2, **COUNTS, "CNN_image_forwards": 0,
                          "GPU_calls": 0, "backward_calls": 0, "optimizer_updates": 0}), flush=True)

    def setUp(self):
        torch.manual_seed(20262020)
        self.bins = 5
        self.log_a = torch.randn(2, 5, dtype=torch.float64).log_softmax(-1)
        self.log_p = torch.randn(2, 5, 5, dtype=torch.float64).log_softmax(-1)
        self.log_b = torch.randn(2, 5, 5, dtype=torch.float64).log_softmax(-1)
        energy = torch.randn(2, 5, 5, dtype=torch.float64)
        self.log_f = energy.flatten(1).log_softmax(-1).reshape_as(energy)
        self.labels = torch.zeros(2, 3, 5, dtype=torch.float64)
        for batch in range(2):
            for role in range(3):
                lower = (batch + role) % 5
                self.labels[batch, role, lower] = .3
                self.labels[batch, role, (lower + 1) % 5] = .7
        self.atoms, self.masks = reading_atoms(self.bins)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)

    def test_enumeration_full_separable_and_zero_factor(self):
        """Each deletion must use true joint conditionals and the same reading atoms."""
        log_sep = self.log_f.logsumexp(-1)[:, :, None] + self.log_f.logsumexp(-2)[:, None]
        zero = torch.full_like(self.log_f, -2 * math.log(self.bins))
        for factor in (self.log_f, log_sep, zero):
            actual = summarize(self.log_a, self.log_p, self.log_b, factor, self.atoms, self.labels, tile=2, atom_masks=self.masks)
            expected = enumerate_conditionals(self.log_a, self.log_p, self.log_b, factor, self.atoms, self.labels, self.masks)
            for key, value in expected.items():
                self.assertClose(actual[key], value)
            self.assertClose(actual["mass_error"], torch.zeros(2, dtype=torch.float64))
            if factor is zero:
                star_readings = torch.einsum("bad,bas,ds->ba", self.log_p.exp(), self.log_b.exp(), self.atoms)
                self.assertClose(actual["A"], self.log_a.exp())
                self.assertClose(actual["conditional_reading"], star_readings)
        zero_span = torch.zeros_like(self.log_f)
        zero_span[:, :, 0] = 1
        self.assertClose((zero_span * self.atoms).sum((1, 2)) / self.bins, torch.full((2,), .5, dtype=torch.float64))

    def test_rotation_offset_and_underflowed_reference(self):
        """An underflowed global A must retain its valid reference-conditional evidence."""
        log_a = self.log_a.clone()
        log_a[:, 0] -= 1200
        log_a = log_a.log_softmax(-1)
        original = summarize(log_a, self.log_p, self.log_b, self.log_f, self.atoms, self.labels, tile=2, atom_masks=self.masks)
        self.assertTrue(bool((original["A"][:, 0] == 0).all()))
        self.assertTrue(bool(torch.isfinite(original["conditional_reading"][:, 0]).all()))
        expected = enumerate_conditionals(log_a, self.log_p, self.log_b, self.log_f, self.atoms, self.labels, self.masks)
        for key, value in expected.items():
            self.assertClose(original[key], value)
        rotated = summarize(log_a.roll(2, -1), self.log_p.roll(2, 1), self.log_b.roll(2, 1),
                            self.log_f.roll((2, 2), (1, 2)), self.atoms, self.labels.roll(2, -1), tile=3, atom_masks=self.masks)
        shifted = summarize(log_a, self.log_p, self.log_b, self.log_f + 19.25, self.atoms, self.labels, tile=2, atom_masks=self.masks)
        for key in expected:
            if original[key].ndim == 1:
                self.assertClose(rotated[key], original[key])
            else:
                self.assertClose(rotated[key], original[key].roll(2, -1))
            self.assertClose(shifted[key], original[key] + (19.25 if key == "logZ_a" else 0))


if __name__ == "__main__":
    unittest.main()
