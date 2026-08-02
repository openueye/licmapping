from __future__ import annotations

import numpy as np
import pytest

from lic_mapping.evaluation import _write_ply


@pytest.mark.parametrize("sh_degree", (0, 3))
def test_write_ply_uses_standard_gaussian_splatting_schema(tmp_path, sh_degree: int) -> None:
    count = 2
    sh_rest = np.zeros((count, (sh_degree + 1) ** 2 - 1, 3), dtype=np.float32)
    path = tmp_path / f"degree{sh_degree}.ply"
    _write_ply(
        path,
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.ones((count, 1, 3), dtype=np.float32),
        sh_rest,
        np.asarray([[0.1], [0.2]], dtype=np.float32),
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    )

    header = path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
    properties = [
        line.rsplit(" ", 1)[1]
        for line in header.splitlines()
        if line.startswith("property float ")
    ]
    expected = ["x", "y", "z", "nx", "ny", "nz"]
    expected += [f"f_dc_{index}" for index in range(3)]
    expected += [f"f_rest_{index}" for index in range(sh_rest.shape[1] * 3)]
    expected += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert properties == expected
