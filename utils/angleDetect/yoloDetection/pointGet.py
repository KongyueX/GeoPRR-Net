
import numpy as np
import math

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

def line_intersection(line1, line2):
    """
    计算两条直线的交点
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # 计算分母
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if denom == 0:  # 直线平行
        return None
    
    # 计算参数
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # 计算交点坐标
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (x, y)

def get_line_border_intersections(pt1, pt2, image_shape):
    """
    计算两点连线与图像边框的交点
    pt1, pt2: 两个点的坐标 (x, y)
    image_shape: 图像形状 (height, width) 或 (height, width, channels)
    """
    h, w = image_shape[:2] if len(image_shape) > 2 else image_shape
    
    # 图像边框的四条边
    borders = [
        (0, 0, w-1, 0),      # 上边
        (w-1, 0, w-1, h-1),  # 右边
        (0, h-1, w-1, h-1),  # 下边
        (0, 0, 0, h-1)       # 左边
    ]
    
    intersections = []
    
    for border in borders:
        intersection = line_intersection((pt1[0], pt1[1], pt2[0], pt2[1]), border)
        if intersection:
            x, y = intersection
            # 检查交点是否在边框线段上
            if (0 <= x <= w-1 and 0 <= y <= h-1 and
                min(border[0], border[2]) - 1 <= x <= max(border[0], border[2]) + 1 and
                min(border[1], border[3]) - 1 <= y <= max(border[1], border[3]) + 1):
                intersections.append((int(round(x)), int(round(y))))
    
    return intersections

def get_perpendicular_line_at_midpoint(pt1, pt2, length=100):
    """
    计算两点连线中点处的垂直线
    pt1, pt2: 两个点的坐标 (x, y)
    length: 垂直线段的长度（从中点向两侧延伸的距离）
    返回: 垂直线段的两个端点
    """
    # 计算中点
    mid_x = (pt1[0] + pt2[0]) / 2
    mid_y = (pt1[1] + pt2[1]) / 2
    
    # 计算原线段的方向向量
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    
    # 计算垂直方向向量
    perp_dx = -dy
    perp_dy = dx
    
    # 归一化垂直向量
    norm = math.sqrt(perp_dx**2 + perp_dy**2)
    if norm > 0:
        perp_dx /= norm
        perp_dy /= norm
    
    # 计算垂直线段的两个端点
    pt3 = (int(mid_x + perp_dx * length/2), int(mid_y + perp_dy * length/2))
    pt4 = (int(mid_x - perp_dx * length/2), int(mid_y - perp_dy * length/2))
    
    return pt3, pt4

# 示例使用
def demonstrate_intersections(pt1, pt2, imgCenter, image):
    # 绘制中点
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2
    mid = (mid_x, mid_y)

    distanceY = imgCenter[1] - mid[1]
    distanceX = imgCenter[0] - mid[0] 

    # 计算与边框的交点 
    intersections = get_line_border_intersections(pt1, pt2, image.shape)
    intersections = ((intersections[0][0], intersections[0][1] + distanceY),
                     (intersections[1][0], intersections[1][1] + distanceY)
    
    )

    # 计算中点垂直线
    perp_pt1, perp_pt2 = get_perpendicular_line_at_midpoint(pt1, pt2, 200)
    
    # 计算垂直线与边框的交点s
    perp_intersections = get_line_border_intersections(perp_pt1, perp_pt2, image.shape)
    perp_intersections = ((perp_intersections[0][0] + distanceX, perp_intersections[0][1]),
                          (perp_intersections[1][0] + distanceX, perp_intersections[1][1])
    
    )


    return {'left': intersections[1], 
            'right': intersections[0], 
            'up': perp_intersections[0], 
            'down': perp_intersections[1]
    
    }

if __name__ == "__main__":

    # 定义两个点
    pt1 = (100, 100)
    pt2 = (500, 300)

    # 创建测试图像
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    mid, left, right, up, down = demonstrate_intersections(pt1, pt2, image)
