"""Render one SyncG group with a fixed scene and controlled pointer states.

Run with Blender's --python option, passing this script's arguments after --.
The vendor source and assets remain external to the project code.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import sys
import types


UPSTREAM_REVISION = "337b868f0f5495b31acc383433cac44040247bf2"


def load_vendor(path: Path) -> types.ModuleType:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    module = types.ModuleType("syncg_state_pilot_vendor")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "name_full"):
        return {"datablock_type": type(value).__name__, "name": value.name_full}
    return [plain(item) for item in value]


def properties(block):
    return {prop.identifier: plain(getattr(block, prop.identifier)) for prop in block.bl_rna.properties
            if not prop.is_readonly and prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}}


def node_tree(tree):
    if tree is None:
        return None
    nodes = []
    for node in sorted(tree.nodes, key=lambda value: value.name):
        record = {"name": node.name, "type": node.bl_idname, "properties": properties(node),
                  "inputs": [{"name": socket.name, "identifier": socket.identifier,
                              "value": plain(socket.default_value)}
                             for socket in node.inputs if hasattr(socket, "default_value")]}
        if getattr(node, "image", None) is not None:
            record["image"] = {"name": node.image.name, "filepath": node.image.filepath,
                               "colorspace": node.image.colorspace_settings.name}
        if hasattr(node, "color_ramp"):
            record["color_ramp"] = {"interpolation": node.color_ramp.interpolation,
                                    "elements": [[element.position, list(element.color)]
                                                 for element in node.color_ramp.elements]}
        nodes.append(record)
    return {"nodes": nodes, "links": sorted([link.from_node.name, link.from_socket.identifier,
                                             link.to_node.name, link.to_socket.identifier]
                                            for link in tree.links)}


def device_records(preferences):
    return [{"name": device.name, "id": device.id, "type": device.type, "use": bool(device.use)}
            for device in preferences.devices]


def configure_devices(bpy, device):
    if device == "CPU":
        return {"backend": "CPU", "devices": []}
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    if not any(device.type == "CUDA" for device in preferences.devices):
        raise RuntimeError("Cycles CUDA device requested but no CUDA device is available")
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    return {"backend": preferences.compute_device_type, "devices": device_records(preferences)}


def scene_factors(bpy, pointer_name, dial_parameters, model_parameters):
    scene = bpy.context.scene
    objects, meshes, texts = [], {}, {}
    for obj in sorted(scene.objects, key=lambda value: value.name):
        record = {"name": obj.name, "type": obj.type, "location": list(obj.location),
                  "scale": list(obj.scale), "rotation_mode": obj.rotation_mode,
                  "hide_render": obj.hide_render, "parent": obj.parent.name if obj.parent else None,
                  "parent_inverse": plain(obj.matrix_parent_inverse),
                  "delta_location": list(obj.delta_location), "delta_scale": list(obj.delta_scale),
                  "delta_rotation_euler": list(obj.delta_rotation_euler),
                  "modifiers": [{"name": modifier.name, "type": modifier.type,
                                 "properties": properties(modifier)} for modifier in obj.modifiers]}
        if obj.name != pointer_name:
            record["rotation_euler"] = list(obj.rotation_euler)
            record["rotation_quaternion"] = list(obj.rotation_quaternion)
            record["rotation_axis_angle"] = list(obj.rotation_axis_angle)
        if obj.data is not None:
            record["data"] = obj.data.name
        objects.append(record)
        if obj.type == "MESH" and obj.data.name not in meshes:
            mesh = obj.data
            meshes[mesh.name] = {"vertices": [list(vertex.co) for vertex in mesh.vertices],
                                 "polygons": [[list(face.vertices), face.material_index, face.use_smooth]
                                              for face in mesh.polygons],
                                 "uv_layers": {layer.name: [list(loop.uv) for loop in layer.data]
                                               for layer in mesh.uv_layers}}
        if obj.type == "FONT":
            texts[obj.name] = {"properties": properties(obj.data),
                               "font": obj.data.font.filepath, "body": obj.data.body}
    fixed_dial = {key: plain(value) for key, value in dial_parameters.items() if key != "pointer_rotate_degree"}
    fixed_model = {key: plain(value) for key, value in model_parameters.items()
                   if key not in {"img_output_path", "use_gpu"}}
    geometry = {"dial_parameters": fixed_dial, "model_parameters": fixed_model,
                "selected_pointer": pointer_name, "objects": objects, "meshes": meshes, "texts": texts}
    appearance = {
        "materials": {material.name: {"properties": properties(material), "nodes": node_tree(material.node_tree)}
                      for material in sorted(bpy.data.materials, key=lambda value: value.name)},
        "material_slots": {obj.name: [slot.material.name if slot.material else None for slot in obj.material_slots]
                           for obj in sorted(scene.objects, key=lambda value: value.name) if hasattr(obj, "material_slots")},
        "world": {"name": scene.world.name, "properties": properties(scene.world),
                  "nodes": node_tree(scene.world.node_tree)} if scene.world else None,
        "lights": {obj.name: properties(obj.data) for obj in scene.objects if obj.type == "LIGHT"},
        "view_settings": properties(scene.view_settings), "display_settings": properties(scene.display_settings),
        "render": {"engine": scene.render.engine, "threads": scene.render.threads,
                   "threads_mode": scene.render.threads_mode, "samples": scene.cycles.samples,
                   "device": scene.cycles.device, "seed": scene.cycles.seed,
                   "animated_seed": scene.cycles.use_animated_seed,
                   "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
                   "quality": scene.render.image_settings.quality,
                   "image_settings": properties(scene.render.image_settings)},
    }
    if scene.cycles.device == "GPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        appearance["render"].update(backend=preferences.compute_device_type, devices=device_records(preferences))
    else:
        appearance["render"].update(backend="CPU", devices=[])
    camera = scene.camera
    camera_parameters = {"name": camera.name, "location": list(camera.location),
                         "rotation_euler": list(camera.rotation_euler), "matrix_world": plain(camera.matrix_world),
                         "data": properties(camera.data), "dof": properties(camera.data.dof)}
    return {"g": geometry, "a": appearance, "c": camera_parameters}


def configure_render(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if args.device == "CUDA" else "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.use_animated_seed = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = args.image_format
    scene.render.image_settings.color_mode = "RGB"
    if args.image_format == "JPEG":
        scene.render.image_settings.quality = 95


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("geometry_diagnostic", "source_training"), default="geometry_diagnostic")
    parser.add_argument("--layout-id")
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--states", type=float, nargs="+", default=[0.15, 0.35, 0.65, 0.85])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--image-format", choices=("PNG", "JPEG"), default="PNG")
    args = parser.parse_args(argv)
    if len(args.states) < 2 or len(set(args.states)) != len(args.states) or any(not 0 < state < 1 for state in args.states):
        parser.error("states must be distinct normalized values strictly between zero and one")
    if min(args.width, args.height, args.threads, args.samples) < 1:
        parser.error("dimensions, threads, and samples must be positive")
    if args.device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import bpy
    import numpy as np

    source = args.source_root.resolve()
    output = args.output.resolve()
    diagnostic = args.role == "geometry_diagnostic"
    factor_path = output / "factors.json.gz"
    extension = "jpg" if args.image_format == "JPEG" else "png"
    if (output / "group.json").exists() or any((output / "images").glob("*.png")) or any((output / "images").glob("*.jpg")):
        raise FileExistsError(f"pilot output already contains results: {output}")
    for folder in ("images", "annotations", "manifests", "asset_shim"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for name, target in {"dial_texture": source / "blender_project/dial_texture", "font_files": source / "font_files"}.items():
        if not target.is_dir():
            raise FileNotFoundError(target)
        link = output / "asset_shim" / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    bpy.ops.wm.open_mainfile(filepath=str(source / "blender_project/fakemeter_oil_new.blend"))
    devices = configure_devices(bpy, args.device)
    vendor = load_vendor(source / "scripts/generate_syncg.py")
    original_projection = vendor.world_to_camera
    vendor.world_to_camera = lambda vector, _width, _height: original_projection(vector, args.width, args.height)
    np.random.seed(args.seed)
    random.seed(args.seed)
    scene_file = vendor.make_random_sense(scene_root=str(source / "scene_file"))
    layout_id = args.layout_id or f"syncg_state_{'pilot' if diagnostic else 'training'}_{args.seed}"
    names = [f"{layout_id}_s{index:02d}" for index in range(len(args.states))]
    dial, model = vendor.parameter_generate(save_folder=str(output / "images"), save_name=names[0],
                                           use_gpu=[], width=args.width, height=args.height)
    dial = {key: plain(value) for key, value in dial.items()}
    model = {key: plain(value) for key, value in model.items()}
    span = dial["long_interval_degree"] * dial["long_num"]
    dial["pointer_rotate_degree"] = args.states[0] * span
    model["img_output_format"] = args.image_format
    model["img_output_path"] = str(output / "images" / f"{names[0]}.{extension}")
    scene = bpy.context.scene
    configure_render(scene, args)
    baseline, checks, failures = None, [], []
    current_state = args.states[0]

    def inspect(phase):
        nonlocal baseline
        actual = scene_factors(bpy, vendor.pointer_name, dial, model)
        if baseline is None:
            baseline = actual
        equal = actual == baseline
        checks.append({"state": current_state, "phase": phase, "nonintervened_equal": equal})
        if not equal:
            failures.append({"state": current_state, "phase": phase})

    def before_render(scene, *_):
        configure_render(scene, args)
        inspect("render_pre")

    def after_render(_scene, *_):
        inspect("render_post")

    bpy.app.handlers.render_pre.append(before_render)
    bpy.app.handlers.render_post.append(after_render)
    provenance = {"upstream_repository": "https://github.com/SKYWOWDYH/SyncG", "upstream_revision": UPSTREAM_REVISION,
                  "vendor_source": vendor.__file__, "blend_file": str(source / "blender_project/fakemeter_oil_new.blend"),
                  "blender_version": bpy.app.version_string, "seed": args.seed, "scene_file": scene_file,
                  "execution": f"CYCLES_{args.device}", "backend": devices["backend"], "devices": devices["devices"],
                  "threads": args.threads, "samples": args.samples,
                  "image_format": args.image_format, "image_quality": scene.render.image_settings.quality,
                  "intervention": "same selected pointer rotation only", "longmark_projection_size": [args.width, args.height],
                  "virtual_q_semantics": "Virtual radius-one dial-plane readout point is derived by GeoPRRV2Dataset; Pointer.outside_kp is the physical pointer tip, not Q."}
    images = []
    try:
        vendor.sync_gauge(**dial, **model, project_root=str(output / "asset_shim"))
        pointer = bpy.data.objects[vendor.pointer_name]
        first_rotation = float(pointer.rotation_euler[2])
        for index, state in enumerate(args.states):
            current_state = state
            degree = state * span
            dial["pointer_rotate_degree"] = degree
            if index:
                pointer.rotation_euler[2] = first_rotation - math.radians((state - args.states[0]) * span)
                bpy.context.view_layer.update()
                inspect("before_render_call")
                if failures:
                    raise RuntimeError(f"non-intervened scene factors changed: {failures}")
                configure_render(scene, args)
                scene.render.filepath = str(output / "images" / f"{names[index]}.{extension}")
                bpy.ops.render.render(write_still=True)
            if baseline is None or failures or any(sum(check["phase"] == phase for check in checks) != index + 1
                                                    for phase in ("render_pre", "render_post")):
                raise RuntimeError(f"render identity checks did not pass: {failures}")
            if not diagnostic and index == 0:
                with gzip.open(factor_path, "wt", encoding="utf-8") as handle:
                    json.dump({"layout_id": layout_id, "fixed_factors": baseline}, handle, allow_nan=False)
            vendor.output_json(**dial, seed=args.seed, width=args.width, height=args.height,
                               save_folder=str(output / "annotations"), save_name=names[index], scene_name=scene_file)
            annotation_path = output / "annotations" / f"{names[index]}.json"
            annotation = json.loads(annotation_path.read_text())
            factors = {**baseline, "r": state} if diagnostic else {"reference": str(factor_path), "r": state}
            annotation.update({"layout_id": layout_id, "factors": factors, "provenance": provenance,
                               "diagnostic_only": diagnostic, "role": args.role,
                               "pointer_rotation_euler": list(pointer.rotation_euler), "normalized_state": state})
            annotation_path.write_text(json.dumps(annotation, allow_nan=False, indent=2) + "\n")
            images.append({"sample_id": names[index], "layout_id": layout_id, "normalized_state": state,
                           "diagnostic_only": diagnostic, "role": args.role,
                           "pointer_rotate_degree": degree, "physical_reading": annotation["ground_truth"],
                           "image_path": str(output / "images" / f"{names[index]}.{extension}"),
                           "annotation_path": str(annotation_path)})
    finally:
        bpy.app.handlers.render_pre.remove(before_render)
        bpy.app.handlers.render_post.remove(after_render)
    (output / "manifests/raw_images.jsonl").write_text("".join(json.dumps(row) + "\n" for row in images))
    group = {"pilot_only": diagnostic, "diagnostic_only": diagnostic, "role": args.role,
             "layout_id": layout_id, "provenance": provenance,
             "fixed_factors": baseline if diagnostic else {"reference": str(factor_path)},
             "states": args.states, "images": images, "identity_checks": checks,
             "annotation_format": "Upstream output_json plus explicit factors; convert with syncg_row before dataset loading."}
    (output / "group.json").write_text(json.dumps(group, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "images": len(images), "nonintervened_equal": not failures}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
