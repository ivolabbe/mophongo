"""`mophongo config <out.json>`: write a run config at its defaults.

The point of the command is that the file it writes is a *usable starting
point*, so the tests check it parses back through the loader and carries the
real defaults, not just that a file appeared.
"""

from __future__ import annotations

import json

import pytest

from mophongo.cli import write_default_config
from mophongo.fit import FitConfig
from mophongo.pipeline import RunConfig


def test_written_config_loads_back(tmp_path):
    out = write_default_config(tmp_path / "default.json")

    cfg = RunConfig.from_json(out)

    # from_json rejects unknown keys, so a clean load is also a check that
    # nothing was dumped that RunConfig would not accept back.
    assert isinstance(cfg, RunConfig)
    assert cfg.name.startswith("<")


def test_every_fit_setting_is_written_out(tmp_path):
    """`fit` defaults to an empty dict, which would document nothing."""
    out = write_default_config(tmp_path / "default.json")
    cfg = RunConfig.from_json(out)

    from dataclasses import fields

    expected = {f.name for f in fields(FitConfig)}
    assert set(cfg.fit) == expected
    assert len(expected) > 20  # guards against a future empty-dict regression


def test_values_are_the_current_defaults(tmp_path):
    """Spot-check settings that changed recently, so a stale dump is visible."""
    out = write_default_config(tmp_path / "default.json")
    cfg = RunConfig.from_json(out)
    default = FitConfig()

    assert cfg.fit["astrom_model"] == default.astrom_model == "poly"
    assert cfg.fit["astrom_robust"] is default.astrom_robust is True
    assert cfg.fit["phot"]["aperture_ee"] == default.phot.aperture_ee == 0.70


def test_required_fields_are_placeholders_not_empty(tmp_path):
    """The nine fields with no default must be visibly unfilled.

    An empty string would look like a legitimate value and fail later, deep in
    a run; an angle-bracket placeholder cannot be mistaken for one.
    """
    out = write_default_config(tmp_path / "default.json")
    cfg = RunConfig.from_json(out)

    for field in ("name", "out_dir", "sci_hi", "segmap", "catalog",
                  "sci_lo", "wht_lo", "csv_hi", "csv_lo"):
        value = getattr(cfg, field)
        assert value.startswith("<") and value.endswith(">"), field


def test_header_is_comments_only(tmp_path):
    """The header must survive `from_json`, which strips '#' lines."""
    out = write_default_config(tmp_path / "default.json")
    text = out.read_text()

    header = list(
        line for line in text.splitlines()[: text.splitlines().index("{")]
    )
    assert header and all(line.startswith("#") for line in header)
    # and the remainder is valid json on its own
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    json.loads(body)


def test_refuses_to_clobber_without_force(tmp_path):
    """Configs get hand-edited after generation; overwriting would lose that."""
    out = tmp_path / "default.json"
    write_default_config(out)
    out.write_text("# edited by hand\n{}\n")

    with pytest.raises(FileExistsError):
        write_default_config(out)

    assert "edited by hand" in out.read_text()

    write_default_config(out, force=True)
    assert "edited by hand" not in out.read_text()


def test_cli_entry_point(tmp_path):
    from mophongo.cli import main

    target = tmp_path / "sub" / "default.json"
    main(["config", str(target)])  # parent directory is created

    assert target.exists()
    assert RunConfig.from_json(target).fit["astrom_robust"] is True
