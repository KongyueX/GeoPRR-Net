"""Conditional span-domain bounds from ordered integer conic observations.

The input contract is the closed preimage of integer truncation, consecutive
equally angular tick identities on one projected circle, and at least five
ticks. These conditions are not established by this module for RGB observations.
The result excludes part of the span domain; it is not a reading certificate.
All domain predicates use integers or Fraction. Floating bounds are display-only.
"""
from fractions import Fraction
from itertools import product
import math


def _coordinate_cell(value):
    if int(value) != value:
        raise ValueError("requires integer truncation labels")
    integer = int(value)
    return (integer, integer+1) if integer > 0 else (integer-1, integer) if integer < 0 else (-1, 1)


def _corners(point):
    x, y = map(_coordinate_cell, point)
    return tuple(product(x, y))


def _triangle_interval(first, second, reference):
    values = [(a[0]-r[0])*(b[1]-r[1])-(a[1]-r[1])*(b[0]-r[0])
              for a, b, r in product(first, second, reference)]
    return min(values), max(values)


def _interval_product(first, second):
    values = [a*b for a, b in product(first, second)]
    return min(values), max(values)


def _determinant_line(first, second):
    return (first[0]*second[1]-first[1]*second[0],
            first[1]-second[1], second[0]-first[0])


def _linear_product(first, second):
    a, b, c = first
    d, e, f = second
    return a*d, a*e+b*d, a*f+c*d, b*e, b*f+c*e, c*f


def _quadratic_rectangle_range(coefficients, rectangle):
    """Exact extrema of a+b*x+c*y+d*x*x+e*x*y+f*y*y on a rectangle."""
    a, b, c, d, e, f = map(Fraction, coefficients)
    x0, y0, x1, y1 = map(Fraction, rectangle)
    points = list(product((x0, x1), (y0, y1)))
    for x in (x0, x1):
        if f:
            y = (-c-e*x)/(2*f)
            if y0 <= y <= y1:
                points.append((x, y))
    for y in (y0, y1):
        if d:
            x = (-b-e*y)/(2*d)
            if x0 <= x <= x1:
                points.append((x, y))
    determinant = 4*d*f-e*e
    if determinant:
        x = (e*c-2*f*b)/determinant
        y = (e*b-2*d*c)/determinant
        if x0 <= x <= x1 and y0 <= y <= y1:
            points.append((x, y))
    values = [a+b*x+c*y+d*x*x+e*x*y+f*y*y for x, y in points]
    return min(values), max(values)


def _rational_pair(value):
    value = Fraction(value)
    return value.numerator, value.denominator


def _span_lower(count, stride, ratio_lower):
    square = (3*ratio_lower-4)/ratio_lower
    scale = 1 << 64
    while square.numerator*scale*scale < square.denominator:
        scale *= 2
    tangent = Fraction(math.isqrt(square.numerator*scale*scale//square.denominator), scale)
    lower = Fraction(2*(count-1), stride)*tangent/(1+tangent*tangent)
    return lower, max(0., math.nextafter(float(lower), -math.inf))


def ordered_tick_span_domain(integer_pixels):
    """Return an exact conditional positive lower bound on span, when proved.

    Ratio fields are exact ``(numerator, denominator)`` pairs. The naive ratio
    interval discards shared-point dependence; effective_R_lower may additionally
    use the shared-reference sign proof for F=3*numerator-4*denominator. A zero
    lower bound reports an unexcluded zero-span limit, not a zero physical span.
    """
    count = len(integer_pixels)
    if count < 5:
        raise ValueError("requires at least five consecutive equal-angle ticks")
    boxes = [_corners(point) for point in integer_pixels]
    stride = (count-1)//4
    indices = tuple(i*stride for i in range(4))
    midpoints = [tuple(sum(point[axis] for point in box)//2 for axis in range(2)) for box in boxes]
    # Twice each cell midpoint is integral, including the wider zero cell.
    reference = max((i for i in range(count) if i not in indices), key=lambda i:
                    min(sum((midpoints[i][axis]-midpoints[k][axis])**2 for axis in range(2)) for k in indices))
    points, ref = [boxes[k] for k in indices], boxes[reference]
    numerator = _interval_product(_triangle_interval(points[0], points[2], ref),
                                  _triangle_interval(points[1], points[3], ref))
    denominator = _interval_product(_triangle_interval(points[0], points[3], ref),
                                    _triangle_interval(points[1], points[2], ref))
    output = {"tick_count": count, "stride": stride, "selected_tick_indices": indices,
              "reference_index": reference, "naive_R_lower": None, "naive_R_upper": None,
              "effective_R_lower": None, "effective_R_upper": None,
              "positive_span_lower_proved_under_cell_model": False, "naive_positive_span_lower": False,
              "rational_span_lower": (0, 1), "span_lower_radians_outward": 0.,
              "dependency_status": "not_attempted", "main_corner_cases_checked": 0, "signed_F_margin": None}
    if denominator[0] <= 0 <= denominator[1]:
        return {**output, "status": "denominator_interval_contains_zero"}
    ratios = [Fraction(n, d) for n, d in product(numerator, denominator)]
    low, high = min(ratios), max(ratios)
    output.update(naive_R_lower=_rational_pair(low), naive_R_upper=_rational_pair(high),
                  effective_R_lower=_rational_pair(low), effective_R_upper=_rational_pair(high))
    if high <= Fraction(4, 3) or low >= 2:
        return {**output, "status": "outer_cross_ratio_excludes_legal_equal_angle_model"}
    if low > Fraction(4, 3):
        output.update(naive_positive_span_lower=True, dependency_status="prior_positive_retained")
    else:
        rectangle = (*map(min, zip(*ref)), *map(max, zip(*ref)))
        margin = None
        positive_denominator = denominator[0] > 0
        for checked, (a, b, c, d) in enumerate(product(*points), start=1):
            first = _linear_product(_determinant_line(a, c), _determinant_line(b, d))
            second = _linear_product(_determinant_line(a, d), _determinant_line(b, c))
            f_low, f_high = _quadratic_rectangle_range(tuple(3*x-4*y for x, y in zip(first, second)), rectangle)
            candidate = f_low if positive_denominator else -f_high
            output["main_corner_cases_checked"] = checked
            if candidate <= 0:
                return {**output, "status": "zero_span_limit_not_excluded", "dependency_status": "strict_sign_not_proved"}
            margin = candidate if margin is None else min(margin, candidate)
        low = Fraction(4, 3)+margin/(3*max(abs(value) for value in denominator))
        output.update(effective_R_lower=_rational_pair(low), signed_F_margin=_rational_pair(margin))
        if low >= 2:
            return {**output, "status": "outer_cross_ratio_excludes_legal_equal_angle_model",
                    "dependency_status": "enclosure_excludes_legal_model"}
        output["dependency_status"] = "strict_sign_proved"
    lower, outward = _span_lower(count, stride, low)
    return {**output, "status": "positive_span_lower", "rational_span_lower": _rational_pair(lower),
            "span_lower_radians_outward": outward,
            "positive_span_lower_proved_under_cell_model": lower > 0}
