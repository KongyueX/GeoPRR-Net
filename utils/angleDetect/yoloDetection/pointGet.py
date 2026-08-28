import math

import numpy as np


def line_intersection(line1, line2):
    """计算两条直线的交点。"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return x, y


def get_line_border_intersections(pt1, pt2, image_shape):
    """计算两点连线与图像边框的交点。"""
    h, w = image_shape[:2] if len(image_shape) > 2 else image_shape
    borders = [
        (0, 0, w - 1, 0),
        (w - 1, 0, w - 1, h - 1),
        (0, h - 1, w - 1, h - 1),
        (0, 0, 0, h - 1),
    ]

    intersections = []
    for border in borders:
        intersection = line_intersection(
            (pt1[0], pt1[1], pt2[0], pt2[1]), border
        )
        if intersection:
            x, y = intersection
            if (
                0 <= x <= w - 1
                and 0 <= y <= h - 1
                and min(border[0], border[2]) - 1
                <= x
                <= max(border[0], border[2]) + 1
                and min(border[1], border[3]) - 1
                <= y
                <= max(border[1], border[3]) + 1
            ):
                intersections.append((int(round(x)), int(round(y))))

    return intersections


def get_perpendicular_line_at_midpoint(pt1, pt2, length=100):
    """计算两点连线中点处的定长垂直线段。"""
    mid_x = (pt1[0] + pt2[0]) / 2
    mid_y = (pt1[1] + pt2[1]) / 2
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    perp_dx = -dy
    perp_dy = dx

    norm = math.sqrt(perp_dx**2 + perp_dy**2)
    if norm > 0:
        perp_dx /= norm
        perp_dy /= norm

    pt3 = (int(mid_x + perp_dx * length / 2), int(mid_y + perp_dy * length / 2))
    pt4 = (int(mid_x - perp_dx * length / 2), int(mid_y - perp_dy * length / 2))
    return pt3, pt4


def demonstrate_intersections(pt1, pt2, img_center, image):
    """Return shifted horizontal and vertical border intersections."""
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2
    mid = (mid_x, mid_y)

    distance_y = img_center[1] - mid[1]
    distance_x = img_center[0] - mid[0]
    intersections = get_line_border_intersections(pt1, pt2, image.shape)
    intersections = (
        (intersections[0][0], intersections[0][1] + distance_y),
        (intersections[1][0], intersections[1][1] + distance_y),
    )

    perp_pt1, perp_pt2 = get_perpendicular_line_at_midpoint(pt1, pt2, 200)
    perp_intersections = get_line_border_intersections(
        perp_pt1, perp_pt2, image.shape
    )
    perp_intersections = (
        (
            perp_intersections[0][0] + distance_x,
            perp_intersections[0][1],
        ),
        (
            perp_intersections[1][0] + distance_x,
            perp_intersections[1][1],
        ),
    )

    return {
        "left": intersections[1],
        "right": intersections[0],
        "up": perp_intersections[0],
        "down": perp_intersections[1],
    }


if __name__ == "__main__":
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    print(
        demonstrate_intersections(
            (100, 100),
            (500, 300),
            (300, 200),
            test_image,
        )
    )
