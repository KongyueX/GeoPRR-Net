"""Render one physical source group with a fixed local optical-axis camera roll."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import render_geoprr_v2_state_group as state_renderer


PROTOCOL = "geoprr_v2_camera_roll_source_v22"


def local_roll_quaternion(orientation, roll_degrees):
    """Compose a wxyz orientation with a rotation about its local optical axis."""
    if not math.isfinite(roll_degrees) or not -180 <= roll_degrees < 180:
        raise ValueError("source camera roll must be finite in[-180,180) degrees")
    w, x, y, z = orientation
    half = math.radians(roll_degrees) / 2
    cosine, sine = math.cos(half), math.sin(half)
    return (w * cosine - z * sine, x * cosine + y * sine,
            y * cosine - x * sine, z * cosine + w * sine)


def apply_camera_roll(bpy, camera_name, roll_degrees):
    from mathutils import Quaternion, Vector

    camera = bpy.data.objects[camera_name]
    if camera.rotation_mode != "XYZ":
        raise ValueError("the pinned camera must retain its existing XYZ rotation mode")
    bpy.context.view_layer.update()
    before = camera.rotation_euler.to_quaternion().normalized()
    before_world = camera.matrix_world.to_quaternion().normalized()
    location, fov = list(camera.location), float(camera.data.angle)
    rolled = Quaternion(local_roll_quaternion(tuple(before), roll_degrees)).normalized()
    camera.rotation_euler = rolled.to_euler("XYZ")
    bpy.context.view_layer.update()
    after_world = camera.matrix_world.to_quaternion().normalized()
    axis = Vector((0., 0., -1.))
    axis_error = max(abs(a - b) for a, b in zip(before_world @ axis, after_world @ axis))
    if axis_error > 1e-6 or list(camera.location) != location or float(camera.data.angle) != fov:
        raise ValueError("local camera roll changed its optical direction, position or FOV")
    return {"camera_name": camera_name, "roll_degrees": roll_degrees,
            "axis": "camera local Z; the optical direction is unchanged",
            "composition": "q_after=q_before*q_local_Z_roll", "rotation_mode": "XYZ",
            "before_quaternion_wxyz": list(before),
            "after_quaternion_wxyz": list(camera.rotation_euler.to_quaternion()),
            "location": location, "fov_radians": fov,
            "optical_axis_max_abs_difference": axis_error}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--states", type=float, nargs=4, required=True)
    parser.add_argument("--roll-degrees", type=float, required=True)
    parser.add_argument("--split", choices=("train", "roll_validation"), required=True)
    parser.add_argument("--role", choices=("source_training",), default="source_training")
    parser.add_argument("--state-policy", choices=("interior", "endpoint_v11"), default="interior")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", choices=("CPU", "CUDA"), default="CUDA")
    parser.add_argument("--image-format", choices=("JPEG",), default="JPEG")
    args = parser.parse_args(argv)
    local_roll_quaternion((1., 0., 0., 0.), args.roll_degrees)
    records = []
    original_loader = state_renderer.load_vendor

    def load_vendor_with_roll(path):
        vendor = original_loader(path)
        original_move_camera = vendor.move_camera

        def move_camera(*positional, **keywords):
            original_move_camera(*positional, **keywords)
            if records:
                raise ValueError("one physical group must configure its camera exactly once")
            name = keywords.get("camera_name", positional[0] if positional else "Camera")
            record = apply_camera_roll(vendor.bpy, name, args.roll_degrees)
            records.append(record)
            write_json(args.output.resolve() / "camera_roll.json", record)

        vendor.move_camera = move_camera
        return vendor

    state_renderer.load_vendor = load_vendor_with_roll
    forwarded = ["--source-root", str(args.source_root), "--output", str(args.output),
                 "--layout-id", args.layout_id, "--seed", str(args.seed), "--states", *map(str, args.states),
                 "--role", args.role, "--state-policy", args.state_policy, "--width", str(args.width),
                 "--height", str(args.height), "--threads", str(args.threads), "--samples", str(args.samples),
                 "--device", args.device, "--image-format", args.image_format]
    try:
        state_renderer.main(forwarded)
    finally:
        state_renderer.load_vendor = original_loader
    if len(records) != 1:
        raise ValueError("the physical render did not record one camera-roll intervention")
    output = args.output.resolve()
    group = json.loads((output / "group.json").read_text())
    declaration = {"protocol": PROTOCOL, "split": args.split, "layout_id": args.layout_id,
                   "render_seed": args.seed, "roll_degrees": args.roll_degrees,
                   "camera_roll_record": str(output / "camera_roll.json")}
    group["provenance"]["source_roll"] = declaration
    group["source_roll"] = declaration
    write_json(output / "group.json", group)
    for index, image in enumerate(group["images"]):
        path = Path(image["annotation_path"])
        annotation = json.loads(path.read_text())
        annotation["provenance"]["source_roll"] = declaration
        annotation["source_roll"] = {**declaration, "state_index": index}
        write_json(path, annotation)
    write_json(output / "roll_group.json", {**declaration, "status": "complete", "stage": "render",
               "states": args.states, "state_policy": args.state_policy, "images": len(group["images"]),
               "camera_roll": records[0], "in_group_nonintervened_factors_equal": all(
                   row["nonintervened_equal"] for row in group["identity_checks"])})
    return 0


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
