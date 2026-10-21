"""Focused exact arithmetic checks for the conditional conic span contractor."""
from fractions import Fraction
import math
import unittest

from experiments.geoprr_v2_conic_interval import (
    _coordinate_cell, _quadratic_rectangle_range, ordered_tick_span_domain,
)


class ConicIntervalTests(unittest.TestCase):
    def test_quadratic_edge_interior_and_singular_extrema(self):
        cases = [((4, 0, -4, 1, 0, 1), (-1, 0, 1, 1), (1, 5)),
                 ((Fraction(17, 72), Fraction(-2, 3), 1, 1, 0, 2), (-1, -1, 1, 1),
                  (0, Fraction(353, 72))),
                 ((Fraction(1, 4), -1, -1, 1, 2, 1), (-1, -1, 1, 1), (0, Fraction(25, 4)))]
        for coefficients, rectangle, expected in cases:
            with self.subTest(coefficients=coefficients):
                self.assertEqual(_quadratic_rectangle_range(coefficients, rectangle), expected)

    def test_closed_integer_truncation_cells(self):
        self.assertEqual(_coordinate_cell(3), (3, 4))
        self.assertEqual(_coordinate_cell(-3), (-4, -3))
        self.assertEqual(_coordinate_cell(0), (-1, 1))
        with self.assertRaises(ValueError):
            _coordinate_cell(.5)
        result = ordered_tick_span_domain([(0, 0)]*5)
        self.assertEqual(result["status"], "denominator_interval_contains_zero")
        self.assertFalse(result["positive_span_lower_proved_under_cell_model"])
        self.assertEqual(result["rational_span_lower"], (0, 1))

    def test_known_conic_span_is_enclosed_after_integer_truncation(self):
        count, span = 9, math.pi
        for offset in ((0, 0), (700, 600)):
            with self.subTest(image_translation=offset):
                pixels = []
                for index in range(count):
                    angle = -index*span/(count-1)
                    x, y = math.cos(angle), math.sin(angle)
                    depth = .1*x-.05*y+1
                    pixels.append((int((200*x+20*y+300)/depth-offset[0]),
                                   int((-10*x+180*y+200)/depth-offset[1])))
                result = ordered_tick_span_domain(pixels)
                self.assertTrue(result["positive_span_lower_proved_under_cell_model"])
                q = result["stride"]
                square = math.cos(q*span/(2*(count-1)))**2
                ratio = Fraction.from_float(4*square/(4*square-1))
                self.assertLessEqual(Fraction(*result["naive_R_lower"]), ratio)
                self.assertGreaterEqual(Fraction(*result["naive_R_upper"]), ratio)
                self.assertLessEqual(Fraction(*result["effective_R_lower"]), ratio)
                self.assertLessEqual(result["span_lower_radians_outward"], span)
                self.assertLessEqual(Fraction.from_float(result["span_lower_radians_outward"]),
                                     Fraction(*result["rational_span_lower"]))


if __name__ == "__main__":
    unittest.main()
