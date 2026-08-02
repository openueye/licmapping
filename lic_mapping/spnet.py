from __future__ import annotations

"""SPNet depth-completion adapters used by the LIC2 mapping loop.

The reference implementation consumes a fixed-size TensorRT engine.  This
module keeps that boundary explicit: mapping can be tested with a callable
completer, while a real run can use either the native engine or SAGE's locked
SPNet Large-CNX source plus ``Large_300.pth`` checkpoint.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from pathlib import PurePosixPath
import sys
import types
from typing import Callable, Protocol

import cv2
import numpy as np
import torch

from .rosbag import BagFrame, SOURCE_SPNET


SPNET_MODEL_ID = "spnet-large-300"
SPNET_WEIGHTS_SHA256 = "e4a3dbaae1aff425c26db6083cb6ea8e1d6f13d719cc567054b2522b3bf84559"
SPNET_SOURCE_ID = "SPNet-Large-CNX"
SPNET_SOURCE_COMMIT = "b836bd044517b33d3737094acd6a1f09c2362f04"
SPNET_SOURCE_TREE_SHA256 = "d7e0ec9012622d788fd2cd0a89d8984ff00fccf2569e6ce2009d055427589518"
SPNET_SOURCE_FILES = (
    "SPNet/Hole_Datasets/gitignore.txt",
    "SPNet/README.md",
    "SPNet/RGBD_Datasets/gitignore.txt",
    "SPNet/Test_Datasets/gitignore.txt",
    "SPNet/checkpoints/gitignore.txt",
    "SPNet/config.py",
    "SPNet/src/_init_.py",
    "SPNet/src/custom_blocks.py",
    "SPNet/src/data_tools.py",
    "SPNet/src/losses.py",
    "SPNet/src/modules.py",
    "SPNet/src/networks.py",
    "SPNet/src/src_main.py",
    "SPNet/src/utils.py",
    "SPNet/test.py",
    "SPNet/test_utils.py",
    "SPNet/train.py",
)
SPNET_LARGE_DIMS = [192, 384, 768, 1536]
SPNET_LARGE_DEPTHS = [3, 3, 27, 3]
SPNET_ALIGNMENT = 32


def default_spnet_weights_path() -> Path:
    return Path(__file__).resolve().parents[2] / "SAGE-models" / "Large_300.pth"


def default_spnet_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "SAGE" / "third_party" / "SPNet"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_verified_source(source_root: Path) -> tuple[Path, dict[str, bytes]]:
    root = Path(source_root).expanduser().resolve()
    contents: dict[str, bytes] = {}
    for label in SPNET_SOURCE_FILES:
        path = root.joinpath(*PurePosixPath(label).parts[1:])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SPNet tracked source is unavailable or symbolic: {path}")
        contents[label] = path.read_bytes()
    digest = hashlib.sha256()
    for label in SPNET_SOURCE_FILES:
        file_digest = hashlib.sha256(contents[label]).hexdigest()
        digest.update(f"{label}\0{file_digest}\n".encode("utf-8"))
    actual = digest.hexdigest()
    if actual != SPNET_SOURCE_TREE_SHA256:
        raise ValueError(
            "SPNet source tree SHA-256 mismatch: "
            f"expected {SPNET_SOURCE_TREE_SHA256}, got {actual} at {root}"
        )
    return root, contents


def _load_spnet_network(source_root: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
    root, contents = _read_verified_source(source_root)
    source_dir = root / "src"
    package_name = f"_lic_spnet_{SPNET_SOURCE_TREE_SHA256[:16]}"
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)
    package = types.ModuleType(package_name)
    package.__path__ = []
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        loaded: dict[str, types.ModuleType] = {}
        for short_name in ("custom_blocks", "modules", "networks"):
            label = f"SPNet/src/{short_name}.py"
            path = source_dir / f"{short_name}.py"
            module_name = f"{package_name}.{short_name}"
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = package_name
            sys.modules[module_name] = module
            exec(compile(contents[label], str(path), "exec", dont_inherit=True), module.__dict__)
            loaded[short_name] = module
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise RuntimeError(f"Cannot import verified SPNet source: {source_dir / 'networks.py'}") from exc
    if Path(loaded["networks"].__file__).resolve() != (source_dir / "networks.py").resolve():
        raise RuntimeError("Loaded SPNet source does not match the verified source tree")
    network = loaded["networks"].V2Net(SPNET_LARGE_DIMS, SPNET_LARGE_DEPTHS, 0.2, "CNX")
    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot load SPNet checkpoint: {weights_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"network"}:
        raise ValueError("SPNet checkpoint must contain exactly one network state")
    try:
        network.load_state_dict(payload["network"], strict=True)
    except Exception as exc:
        raise RuntimeError(f"SPNet checkpoint state is incompatible: {weights_path}") from exc
    return network.to(device=device, dtype=torch.float32).eval()


class DepthCompleter(Protocol):
    """Complete a metric RGB-D frame at the configured rasterizer resolution."""

    def complete(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Return a float32 metric depth image with the same HxW shape."""


class CallableDepthCompleter:
    """Small adapter for tests and already-loaded SPNet networks."""

    def __init__(self, predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], *, depth_scale_m: float = 200.0, device: torch.device | str = "cuda") -> None:
        if depth_scale_m <= 0 or not np.isfinite(depth_scale_m):
            raise ValueError("depth_scale_m must be positive and finite")
        self.predictor = predictor
        self.depth_scale_m = float(depth_scale_m)
        self.device = torch.device(device)

    def complete(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        rgb_array, depth_array = _validate_inputs(rgb, depth_m)
        height, width = depth_array.shape
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)
        mask_tensor = (depth_tensor > 0).to(torch.float32)
        with torch.inference_mode():
            prediction = self.predictor(
                rgb_tensor.to(self.device, dtype=torch.float32),
                (depth_tensor / self.depth_scale_m).to(self.device, dtype=torch.float32),
                mask_tensor.to(self.device, dtype=torch.float32),
            )
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, 1, height, width):
            raise ValueError("SPNet predictor must return [1, 1, H, W]")
        result = prediction.detach().to(device="cpu", dtype=torch.float32)[0, 0].numpy()
        result = result * self.depth_scale_m
        if not np.isfinite(result).all():
            raise FloatingPointError("SPNet prediction contains non-finite values")
        return result.astype(np.float32, copy=False)


class TorchScriptDepthCompleter(CallableDepthCompleter):
    """Run a TorchScript-exported SPNet with the LIC2 three-input contract."""

    def __init__(self, model_path: Path, *, depth_scale_m: float = 200.0, device: torch.device | str = "cuda") -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SPNet TorchScript model does not exist: {path}")
        target = torch.device(device)
        try:
            network = torch.jit.load(str(path), map_location=target).eval()
        except Exception as exc:
            raise RuntimeError(f"Cannot load SPNet TorchScript model: {path}") from exc
        super().__init__(network, depth_scale_m=depth_scale_m, device=target)
        self.model_path = path

    def complete(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        return _complete_padded(self.predictor, rgb, depth_m, self.depth_scale_m, self.device)


class SPNetDepthCompleter:
    """Load SAGE's verified SPNet Large-CNX checkpoint directly from ``.pth``."""

    def __init__(
        self,
        weights_path: Path | None = None,
        *,
        source_root: Path | None = None,
        depth_scale_m: float = 200.0,
        device: torch.device | str = "cuda",
    ) -> None:
        path = Path(weights_path or default_spnet_weights_path()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SPNet Large-CNX weights do not exist: {path}")
        actual = _sha256_file(path)
        if actual != SPNET_WEIGHTS_SHA256:
            raise ValueError(
                "SPNet Large-CNX weights SHA-256 mismatch: "
                f"expected {SPNET_WEIGHTS_SHA256}, got {actual} at {path}"
            )
        if depth_scale_m <= 0 or not np.isfinite(depth_scale_m):
            raise ValueError("depth_scale_m must be positive and finite")
        target = torch.device(device)
        self.weights_path = path
        self.source_root, _ = _read_verified_source(source_root or default_spnet_source_root())
        self.source_id = SPNET_SOURCE_ID
        self.source_commit = SPNET_SOURCE_COMMIT
        self.model_id = SPNET_MODEL_ID
        self.weights_sha256 = actual
        self.source_tree_sha256 = SPNET_SOURCE_TREE_SHA256
        self.depth_scale_m = float(depth_scale_m)
        self.device = target
        self.predictor = _load_spnet_network(self.source_root, path, target)

    def complete(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        return _complete_padded(self.predictor, rgb, depth_m, self.depth_scale_m, self.device)


def _complete_padded(
    predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    rgb: np.ndarray,
    depth_m: np.ndarray,
    depth_scale_m: float,
    device: torch.device,
) -> np.ndarray:
    rgb_array, depth_array = _validate_inputs(rgb, depth_m)
    height, width = depth_array.shape
    network_height = ((height + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT
    network_width = ((width + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT
    rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)
    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)
    mask_tensor = (depth_tensor > 0).to(torch.float32)
    pad = (0, network_width - width, 0, network_height - height)
    rgb_tensor = torch.nn.functional.pad(rgb_tensor, pad, mode="replicate")
    depth_tensor = torch.nn.functional.pad(depth_tensor, pad, mode="constant", value=0.0)
    mask_tensor = torch.nn.functional.pad(mask_tensor, pad, mode="constant", value=0.0)
    with torch.inference_mode():
        prediction = predictor(
            rgb_tensor.to(device, dtype=torch.float32),
            (depth_tensor / depth_scale_m).to(device, dtype=torch.float32),
            mask_tensor.to(device, dtype=torch.float32),
        )
    expected = (1, 1, network_height, network_width)
    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected:
        raise ValueError(f"SPNet predictor must return {expected}")
    result = prediction.detach().to(device="cpu", dtype=torch.float32)[0, 0, :height, :width].numpy()
    result = result * depth_scale_m
    if not np.isfinite(result).all():
        raise FloatingPointError("SPNet prediction contains non-finite values")
    return result.astype(np.float32, copy=False)


class TensorRTDepthCompleter:
    """Use the native LIC2 TensorRT SPNet engine.

    LIC2's engine has three input tensors (RGB, scaled sparse depth, mask) and
    one output tensor.  Tensor addresses are bound directly to PyTorch CUDA
    tensors, avoiding a pycuda dependency and keeping the output compatible
    with the rest of the training environment.
    """

    def __init__(self, engine_path: Path, *, width: int, height: int, device: torch.device | str = "cuda", depth_scale_m: float = 200.0) -> None:
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SPNet TensorRT engine does not exist: {path}")
        if width < 1 or height < 1:
            raise ValueError("SPNet engine dimensions must be positive")
        if depth_scale_m <= 0 or not np.isfinite(depth_scale_m):
            raise ValueError("depth_scale_m must be positive and finite")
        if torch.device(device).type != "cuda":
            raise ValueError("TensorRT SPNet requires a CUDA device")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT is required for --spnet-engine") from exc
        target = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        try:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"Cannot deserialize SPNet TensorRT engine: {path}") from exc
        if engine is None:
            raise RuntimeError(f"TensorRT returned no engine for: {path}")
        self._trt = trt
        self._runtime = runtime
        self._engine = engine
        self._context = engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Cannot create TensorRT SPNet execution context")
        self.width = int(width)
        self.height = int(height)
        self.device = target
        self.depth_scale_m = float(depth_scale_m)
        self._inputs = []
        self._outputs = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            (self._inputs if mode == trt.TensorIOMode.INPUT else self._outputs).append(name)
        if len(self._inputs) != 3 or len(self._outputs) != 1:
            raise ValueError("LIC2 SPNet engine must expose exactly 3 inputs and 1 output")

    def complete(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        rgb_array, depth_array = _validate_inputs(rgb, depth_m)
        if rgb_array.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"SPNet engine expects {(self.height, self.width)}, got {rgb_array.shape[:2]}"
            )
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = (depth_tensor > 0).to(torch.float32)
        inputs = (rgb_tensor, depth_tensor / self.depth_scale_m, mask_tensor)
        for name, tensor in zip(self._inputs, inputs):
            self._context.set_input_shape(name, tuple(tensor.shape))
        output_shape = tuple(self._context.get_tensor_shape(self._outputs[0]))
        if any(value < 1 for value in output_shape):
            raise ValueError(f"SPNet engine returned an unresolved output shape: {output_shape}")
        output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        for name, tensor in zip(self._inputs, inputs):
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._context.set_tensor_address(self._outputs[0], int(output.data_ptr()))
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not self._context.execute_async_v3(stream):
            raise RuntimeError("SPNet TensorRT enqueue failed")
        torch.cuda.current_stream(self.device).synchronize()
        if tuple(output.shape) != (1, 1, self.height, self.width):
            raise ValueError(f"SPNet output shape must be [1, 1, H, W], got {tuple(output.shape)}")
        result = output[0, 0].detach().cpu().numpy() * self.depth_scale_m
        if not np.isfinite(result).all():
            raise FloatingPointError("SPNet TensorRT output contains non-finite values")
        return result.astype(np.float32, copy=False)


@dataclass(frozen=True)
class DepthCompletionResult:
    points_world: np.ndarray
    colors: np.ndarray
    depths_m: np.ndarray
    mean_known_bias_m: float | None
    candidate_count: int


def complete_keyframe_points(
    frame: BagFrame,
    completer: DepthCompleter,
    *,
    patch_size: int = 10,
    max_depth_m: float = 20.0,
    edge_threshold: float = 0.1,
) -> DepthCompletionResult:
    """Apply LIC2's sparse SPNet-to-Gaussian selection to one keyframe."""

    if patch_size < 1 or max_depth_m <= 0 or edge_threshold <= 0:
        raise ValueError("SPNet completion settings must be positive")
    raw = np.asarray(frame.center_depth_m, dtype=np.float32)
    completed = np.asarray(completer.complete(frame.rgb, raw), dtype=np.float32)
    if completed.shape != raw.shape or not np.isfinite(completed).all():
        raise ValueError("SPNet completed depth must be finite and match the frame grid")
    known = raw > 0
    mean_bias = float((completed[known] - raw[known]).mean()) if bool(known.any()) else None
    if mean_bias is None or abs(mean_bias) >= 0.1:
        return _empty_completion(mean_bias)

    gradient_x = cv2.Sobel(completed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(completed, cv2.CV_32F, 0, 1, ksize=3)
    not_edge = np.hypot(gradient_x, gradient_y) < edge_threshold
    wanted = (completed - mean_bias > 0) & not_edge
    selected: list[tuple[int, int]] = []
    height, width = raw.shape
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)
            if known[top:bottom, left:right].any():
                continue
            patch = np.where(wanted[top:bottom, left:right], completed[top:bottom, left:right] - mean_bias, np.inf)
            if np.isfinite(patch).any():
                local_y, local_x = np.unravel_index(int(np.argmin(patch)), patch.shape)
                selected.append((top + int(local_y), left + int(local_x)))

    if not selected:
        return _empty_completion(mean_bias)
    pixels = np.asarray(selected, dtype=np.int64)
    depths = (completed[pixels[:, 0], pixels[:, 1]] - mean_bias).astype(np.float32)
    keep = (depths > 0) & (depths <= max_depth_m)
    pixels = pixels[keep]
    depths = depths[keep]
    if not len(pixels):
        return _empty_completion(mean_bias, candidate_count=len(selected))
    u = pixels[:, 1].astype(np.float32)
    v = pixels[:, 0].astype(np.float32)
    camera_points = np.column_stack((
        (u - frame.intrinsics.cx) * depths / frame.intrinsics.fx,
        (v - frame.intrinsics.cy) * depths / frame.intrinsics.fy,
        depths,
    )).astype(np.float32)
    pose = np.asarray(frame.world_from_camera, dtype=np.float32)
    points_world = (camera_points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    colors = frame.rgb[pixels[:, 0], pixels[:, 1]].astype(np.float32, copy=True)
    return DepthCompletionResult(points_world, colors, depths, mean_bias, len(selected))


def _empty_completion(mean_bias: float | None, *, candidate_count: int = 0) -> DepthCompletionResult:
    return DepthCompletionResult(
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        mean_bias,
        candidate_count,
    )


def _validate_inputs(rgb: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth_m)
    if rgb_array.dtype != np.float32 or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("SPNet RGB input must be float32 HxWx3")
    if depth_array.dtype != np.float32 or depth_array.ndim != 2 or depth_array.shape != rgb_array.shape[:2]:
        raise ValueError("SPNet depth input must be float32 HxW matching RGB")
    if not np.isfinite(rgb_array).all() or ((rgb_array < 0) | (rgb_array > 1)).any():
        raise ValueError("SPNet RGB input must be finite and within [0, 1]")
    if not np.isfinite(depth_array).all() or (depth_array < 0).any():
        raise ValueError("SPNet depth input must be finite and non-negative")
    return rgb_array, depth_array


__all__ = [
    "CallableDepthCompleter",
    "DepthCompleter",
    "DepthCompletionResult",
    "SPNetDepthCompleter",
    "TensorRTDepthCompleter",
    "TorchScriptDepthCompleter",
    "complete_keyframe_points",
    "default_spnet_source_root",
    "default_spnet_weights_path",
]
