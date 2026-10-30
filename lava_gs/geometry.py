"""Calibrated terrain settings and asset locations for the lava scenes."""

from physstudio.paths import ROOT, data_path

PRESETS = {
    "lands_end": dict(
        path=str(data_path("lava_gs", "lands_end")),
        config=str(ROOT / "lava_gs/configs/lands_end.json"),
        scale=0.45,
        cameras=[0, 1],
        bounds=[-34.0, -30.0, 30.0, 25.0],
        dx=0.16,
        height=8.0,
        terrain_z_bounds=[-2.0, 16.0],
        collider="terrain",
    ),
    "calico": dict(
        path=str(data_path("lava_gs", "calico")),
        config=str(ROOT / "lava_gs/configs/calico.json"),
        scale=3.0,
        cameras=[0, 1],
        bounds=[-3.0, -2.5, 2.8, 4.0],
        dx=0.16,
        height=8.0,
        terrain_z_bounds=[0.5, 5.0],
        collider="terrain",
    ),
    "zohar": dict(
        path=str(data_path("lava_gs", "zohar")),
        config=str(ROOT / "lava_gs/configs/zohar.json"),
        scale=3.0,
        cameras=[0, 1],
        bounds=[-1.7, -4.2, 1.7, 3.2],
        dx=0.16,
        height=8.0,
        collider="terrain",
    ),
}
