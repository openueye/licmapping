from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = (
    ROOT.parents[1]
    / "06_Referrance"
    / "Gaussian-LIC"
    / "src"
)
REFERENCE_SRC = Path(
    os.environ.get("LIC_REFERENCE_SRC", str(DEFAULT_REFERENCE))
).expanduser().resolve()

if not (REFERENCE_SRC / "rasterizer" / "rasterizer.cpp").is_file():
    raise RuntimeError(
        "LIC reference source is missing. Set LIC_REFERENCE_SRC to "
        "06_Referrance/Gaussian-LIC/src."
    )

RASTERIZER_SRC = REFERENCE_SRC / "rasterizer"
CUDA_SRC = RASTERIZER_SRC / "cuda_rasterizer"


def _glm_include() -> str:
    configured = os.environ.get("LIC_GLM_INCLUDE")
    candidates = [Path(configured)] if configured else []
    gsplat_spec = importlib.util.find_spec("gsplat")
    if gsplat_spec is not None and gsplat_spec.origin is not None:
        candidates.append(
            Path(gsplat_spec.origin).resolve().parent
            / "cuda"
            / "csrc"
            / "third_party"
            / "glm"
        )
    candidates.extend(
        [
            Path(sys.prefix) / "include",
            REFERENCE_SRC.parent.parent / "third_party" / "glm",
            Path("/usr/include"),
            Path("/usr/local/include"),
        ]
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "glm" / "glm.hpp").is_file():
            return str(candidate)
    raise RuntimeError(
        "GLM headers are missing. Set LIC_GLM_INCLUDE to a directory "
        "containing glm/glm.hpp."
    )

sources = [
    str(ROOT / "lic_mapping" / "_binding.cpp"),
    str(RASTERIZER_SRC / "rasterizer.cpp"),
    str(RASTERIZER_SRC / "rasterize_points.cu"),
    str(CUDA_SRC / "adam.cu"),
    str(CUDA_SRC / "forward.cu"),
    str(CUDA_SRC / "backward.cu"),
    str(CUDA_SRC / "rasterizer_impl.cu"),
]

setup(
    name="lic-mapping",
    version="0.1.0",
    description="Python adapter for the Gaussian-LIC CUDA rasterizer",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="lic_mapping._C",
            sources=sources,
            include_dirs=[
                str(REFERENCE_SRC),
                str(RASTERIZER_SRC),
                str(CUDA_SRC),
                _glm_include(),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
