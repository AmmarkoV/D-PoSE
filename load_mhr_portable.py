#!/usr/bin/env python3
from typing import Any
import numpy as np
import torch


def restore_obj(x: Any) -> Any:
    if isinstance(x, dict):
        kind = x.get("__kind__")

        if kind == "torch_tensor":
            return x["data"]

        if kind == "numpy_array":
            return x["data"]

        if kind == "numpy_scalar":
            return x["value"]

        if kind == "float_special":
            val = x["value"]
            if val == "nan":
                return float("nan")
            if val == "inf":
                return float("inf")
            if val == "-inf":
                return float("-inf")

        if kind == "dict":
            return {k: restore_obj(v) for k, v in x["items"].items()}

        if kind == "list":
            return [restore_obj(v) for v in x["items"]]

        if kind == "tuple":
            return tuple(restore_obj(v) for v in x["items"])

        if kind == "set":
            return set(restore_obj(v) for v in x["items"])

        if kind in {"object", "nn_module"}:
            return {
                "__type__": x.get("__type__"),
                "attrs": restore_obj(x.get("attrs", {})),
                "parameters": restore_obj(x.get("parameters", {})),
                "buffers": restore_obj(x.get("buffers", {})),
                "children": restore_obj(x.get("children", {})),
            }

        if kind == "reference":
            return {"__reference__": x["__ref_path__"]}

        if kind in {"repr", "error", "max_depth_reached"}:
            return x

        return {k: restore_obj(v) for k, v in x.items()}

    if isinstance(x, list):
        return [restore_obj(v) for v in x]

    return x


def load_portable(path: str) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return restore_obj(raw)


if __name__ == "__main__":
    data = load_portable("mhr_portable_dump/mhr_dump_lod1.pt")
    print(type(data))
    print(data.keys())

    # examples
    interesting = data.get("interesting", {})
    print("interesting keys:", interesting.keys())

    faces = interesting.get("character.mesh.faces")
    if isinstance(faces, torch.Tensor):
        print("faces tensor:", faces.shape, faces.dtype)
    elif isinstance(faces, np.ndarray):
        print("faces ndarray:", faces.shape, faces.dtype)
    else:
        print("faces not found in interesting")
