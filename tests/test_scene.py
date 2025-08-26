import numpy as np
from mophongo.scene import Scene, build_overlap_graph, partition_scenes
from mophongo.fitter_scene import SceneFitter
from mophongo.templates import Template


def _make_template(data: np.ndarray, label: int) -> Template:
    ny, nx = data.shape
    return Template(data, position=(nx / 2, ny / 2), size=data.shape, label=label)


def test_scene_solve_single_template():
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    tmpl = _make_template(img, 0)
    scene = Scene(id=0, templates=[tmpl], bbox=(0, 2, 0, 2))
    scene.set_band(img, weight=np.ones_like(img))
    fitter = SceneFitter()
    sol = scene.solve(fitter)
    assert np.allclose(sol.flux, [1.0])
    res = scene.residual()
    assert np.allclose(res, np.zeros_like(img))


def test_overlap_graph_partition():
    img = np.zeros((5, 5))
    t1 = Template(img, position=(1, 1), size=(3, 3), label=0)
    t2 = Template(img, position=(3, 1), size=(3, 3), label=1)  # overlaps with t1
    t3 = Template(img, position=(4, 4), size=(1, 1), label=2)  # isolated
    adj = build_overlap_graph([t1, t2, t3])
    groups = partition_scenes(adj)
    assert sorted(groups, key=len) == [[2], [0, 1]]
