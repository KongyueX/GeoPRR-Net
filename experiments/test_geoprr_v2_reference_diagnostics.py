"""Small-PMF arithmetic checks for the frozen best57 diagnostic."""
import argparse
import json
import math
from pathlib import Path
import time
import unittest

import torch

from experiments.geoprr_v2_reference_diagnostics import diagnose_state, role_interventions
from experiments.geoprr_v2_reference_relation import CircularProgressPushforward, circular_two_hot_targets


class ReferenceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)
        self.bins = 8
        self.readout = CircularProgressPushforward(self.bins, progress_bins=17)
        self.state = {"A": torch.rand(2, self.bins),
                      "P_rel": torch.rand(2, self.bins, self.bins),
                      "B_rel": torch.rand(2, self.bins, self.bins)}
        self.state = {key: value/value.sum(-1, keepdim=True) for key, value in self.state.items()}
        self.labels = circular_two_hot_targets(torch.tensor([[.3, 1.1, 3.9], [6.1, .9, 5.4]]), self.bins)

    def test_reference_means_endpoint_mass_and_signed_contributions(self):
        result = diagnose_state(self.state, self.readout, self.labels)
        a, p, b = (self.state[key] for key in ("A", "P_rel", "B_rel"))
        span = (a[:, :, None]*b).sum(1)
        expected_contributions = torch.zeros_like(a)
        for image in range(len(a)):
            for reference in range(self.bins):
                atoms = p[image, reference, :, None]*b[image, reference, None]
                output = self.readout(atoms[None])
                product_atoms = p[image, reference, :, None]*span[image, None]
                product_output = self.readout(product_atoms[None])
                expected_contributions[image, reference] = a[image, reference]*(output["reading"][0]
                                                                                 -product_output["reading"][0])
                self.assertAlmostEqual(float(result["conditional_reading"][image, reference]),
                                       float(output["reading"][0]), places=6)
                self.assertAlmostEqual(float(result["conditional_invalid_mass"][image, reference]),
                                       float(output["invalid_mass"][0]), places=6)
                low, high = 0., 0.
                for delta in range(self.bins):
                    for span_index in range(1, self.bins):
                        d, s = delta*2*math.pi/self.bins, span_index*2*math.pi/self.bins
                        raw = ((d-s/2+math.pi) % (2*math.pi)-math.pi+s/2)/s
                        low += float(atoms[delta, span_index])*int(raw < 0)
                        high += float(atoms[delta, span_index])*int(raw > 1)
                self.assertAlmostEqual(float(result["conditional_outside_low_mass"][image, reference]), low, places=6)
                self.assertAlmostEqual(float(result["conditional_outside_high_mass"][image, reference]), high, places=6)
        torch.testing.assert_close(result["signed_reading_contribution"], expected_contributions, atol=1e-7, rtol=0)
        for key in ("conditional_reading_residual", "signed_contribution_residual"):
            self.assertLess(float(result[key].abs().max()), 1e-6)
        joint = p.transpose(1, 2) @ (a[:, :, None]*b)
        product = (a[:, :, None]*p).sum(1)[:, :, None]*span[:, None]
        torch.testing.assert_close(result["progress_joint"], self.readout(joint)["distribution"])
        torch.testing.assert_close(result["progress_product"], self.readout(product)["distribution"])

    def test_shifted_teacher_support_and_role_interventions(self):
        result = diagnose_state(self.state, self.readout, self.labels)
        interventions = role_interventions(self.state, self.readout, self.labels)
        a, p, b = (self.state[key] for key in ("A", "P_rel", "B_rel"))
        teacher_p, teacher_b = torch.zeros_like(p), torch.zeros_like(b)
        for image in range(len(a)):
            for reference in range(self.bins):
                p_mass, b_mass = 0., 0.
                for relative in range(self.bins):
                    absolute = (reference+relative) % self.bins
                    teacher_p[image, reference, relative] = self.labels[image, 1, absolute]
                    teacher_b[image, reference, relative] = self.labels[image, 2, absolute]
                    p_mass += float(p[image, reference, relative])*int(self.labels[image, 1, absolute] > 0)
                    b_mass += float(b[image, reference, relative])*int(self.labels[image, 2, absolute] > 0)
                self.assertAlmostEqual(float(result["conditional_P_correct_mass"][image, reference]), p_mass, places=6)
                self.assertAlmostEqual(float(result["conditional_B_correct_mass"][image, reference]), b_mass, places=6)
        settings = {"A": (self.labels[:, 0], p, b), "P": (a, teacher_p, b),
                    "B": (a, p, teacher_b), "PB": (a, teacher_p, teacher_b),
                    "AP": (self.labels[:, 0], teacher_p, b), "AB": (self.labels[:, 0], p, teacher_b),
                    "APB": (self.labels[:, 0], teacher_p, teacher_b)}
        for name, (weights, pointer, end) in settings.items():
            atoms = torch.zeros(len(a), self.bins, self.bins)
            for reference in range(self.bins):
                atoms += weights[:, reference, None, None]*pointer[:, reference, :, None]*end[:, reference, None]
            expected = self.readout(atoms)
            for key, value in expected.items():
                torch.testing.assert_close(interventions[name][key], value, atol=2e-7, rtol=0)
            torch.testing.assert_close(interventions[name]["distribution"].sum(-1), torch.ones(len(a)), atol=2e-7, rtol=0)

    def test_teacher_reference_weighting_and_ground_truth_pair(self):
        bins = self.bins
        labels = torch.zeros(1, 3, bins)
        labels[0, 0, 1], labels[0, 0, 2] = .25, .75
        labels[0, 1, 3], labels[0, 2, 5] = 1., 1.
        state = {"A": torch.zeros(1, bins), "P_rel": torch.zeros(1, bins, bins),
                 "B_rel": torch.zeros(1, bins, bins)}
        state["A"][0, 1] = 1.
        state["P_rel"][:, :, 1] = 1.
        state["B_rel"][:, :, 4] = 1.
        state["P_rel"][0, 2].zero_()
        state["P_rel"][0, 2, 3] = 1.
        output = role_interventions(state, self.readout, labels)
        conditional = diagnose_state(state, self.readout, labels)["conditional_reading"]
        expected_a = .25*conditional[0, 1]+.75*conditional[0, 2]
        self.assertAlmostEqual(float(output["A"]["reading"][0]), float(expected_a), places=6)
        self.assertAlmostEqual(float(output["PB"]["reading"][0]), .5, places=6)
        self.assertAlmostEqual(float(output["PB"]["invalid_mass"][0]), 0., places=7)
        self.assertAlmostEqual(float(output["PB"]["outside_arc_mass"][0]), 0., places=7)
        quantized = torch.zeros(1, bins, bins)
        for reference in range(bins):
            for pointer in range(bins):
                for end in range(bins):
                    quantized[0, (pointer-reference) % bins, (end-reference) % bins] += (
                        labels[0, 0, reference]*labels[0, 1, pointer]*labels[0, 2, end])
        expected = self.readout(quantized)
        for key, value in expected.items():
            torch.testing.assert_close(output["APB"][key], value, atol=1e-7, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceDiagnosticTests))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "wall_seconds": time.perf_counter()-started, "checkpoint_loads": 0,
              "CNN_image_forwards": 0, "optimizer_updates": 0, "GPU_calls": 0}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
