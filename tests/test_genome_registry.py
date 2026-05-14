"""Tests for pymisha.genome.registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pymisha.genome.registry import (
    _BUNDLED,
    _load_yaml,
    _normalize_recipe,
    _resolve_genome,
    _validate_recipe,
)

# Expected genome names in the bundled recipes.yaml (mirrors R misha 5.6.16).
BUNDLED_NAMES = {
    "hg19", "hg38", "mm9", "mm10", "mm39", "rn6", "rn7",
    "dm6", "ce11", "sacCer3", "danRer11",
}


def test_bundled_yaml_loads_11_entries():
    entries = _load_yaml(_BUNDLED)
    assert set(entries.keys()) == BUNDLED_NAMES
    assert len(entries) == 11


def test_resolve_hg38_is_ucsc():
    recipe = _resolve_genome("hg38")
    assert recipe["source"] == "ucsc"
    assert recipe["assembly"] == "hg38"
    # _layer annotation should point at the bundled file.
    assert recipe["_layer"] == str(_BUNDLED)


def test_resolve_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not in any registry layer"):
        _resolve_genome("no_such_genome_xyz")


def test_explicit_registry_overrides_bundled(tmp_path: Path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "version: 1\n"
        "genome:\n"
        "  hg38: {source: local, path: /tmp/fake_hg38}\n",
        encoding="utf-8",
    )
    recipe = _resolve_genome("hg38", registry=str(custom))
    assert recipe["source"] == "local"
    assert recipe["path"] == "/tmp/fake_hg38"
    assert recipe["_layer"] == str(custom)


def test_explicit_registry_missing_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        _resolve_genome("hg38", registry=str(missing))


def test_env_registry_honored_when_explicit_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_reg = tmp_path / "env.yaml"
    env_reg.write_text(
        "version: 1\n"
        "genome:\n"
        "  hg38: {source: s3, name: hg38_custom}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYMISHA_GENOME_REGISTRY", str(env_reg))
    # Chdir to a directory with no misha.yaml so project layer is skipped.
    monkeypatch.chdir(tmp_path)
    recipe = _resolve_genome("hg38")
    assert recipe["source"] == "s3"
    assert recipe["name"] == "hg38_custom"
    assert recipe["_layer"] == str(env_reg)


def test_project_misha_yaml_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    proj = tmp_path / "misha.yaml"
    proj.write_text(
        "version: 1\n"
        "genome:\n"
        "  hg38: {source: manual, content: 'ACGT'}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PYMISHA_GENOME_REGISTRY", raising=False)
    monkeypatch.chdir(tmp_path)
    recipe = _resolve_genome("hg38")
    assert recipe["source"] == "manual"
    assert recipe["content"] == "ACGT"
    assert recipe["_layer"] == str(proj)


def test_explicit_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Explicit registry arg must override $PYMISHA_GENOME_REGISTRY."""
    explicit = tmp_path / "explicit.yaml"
    env_reg = tmp_path / "env.yaml"
    explicit.write_text(
        "version: 1\ngenome:\n  hg38: {source: local, path: /from/explicit}\n",
        encoding="utf-8",
    )
    env_reg.write_text(
        "version: 1\ngenome:\n  hg38: {source: local, path: /from/env}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYMISHA_GENOME_REGISTRY", str(env_reg))
    monkeypatch.chdir(tmp_path)  # no misha.yaml here
    recipe = _resolve_genome("hg38", registry=str(explicit))
    assert recipe["path"] == "/from/explicit"
    assert recipe["_layer"] == str(explicit)


def test_env_beats_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """$PYMISHA_GENOME_REGISTRY must override ./misha.yaml when no explicit arg."""
    env_reg = tmp_path / "env.yaml"
    proj = tmp_path / "misha.yaml"
    env_reg.write_text(
        "version: 1\ngenome:\n  hg38: {source: local, path: /from/env}\n",
        encoding="utf-8",
    )
    proj.write_text(
        "version: 1\ngenome:\n  hg38: {source: local, path: /from/proj}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYMISHA_GENOME_REGISTRY", str(env_reg))
    monkeypatch.chdir(tmp_path)
    recipe = _resolve_genome("hg38")
    assert recipe["path"] == "/from/env"
    assert recipe["_layer"] == str(env_reg)


def test_project_beats_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """./misha.yaml must override the bundled recipes.yaml entry for the same name."""
    proj = tmp_path / "misha.yaml"
    proj.write_text(
        "version: 1\ngenome:\n  hg38: {source: local, path: /from/proj}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PYMISHA_GENOME_REGISTRY", raising=False)
    monkeypatch.chdir(tmp_path)
    recipe = _resolve_genome("hg38")
    # Bundled hg38 has source=ucsc; project layer must shadow it.
    assert recipe["source"] == "local"
    assert recipe["path"] == "/from/proj"
    assert recipe["_layer"] == str(proj)


def test_bare_string_shorthand_is_local(tmp_path: Path):
    custom = tmp_path / "shorthand.yaml"
    custom.write_text(
        "version: 1\n"
        "genome:\n"
        "  my_db: /data/groots/my_db\n",
        encoding="utf-8",
    )
    recipe = _resolve_genome("my_db", registry=str(custom))
    assert recipe["source"] == "local"
    assert recipe["path"] == "/data/groots/my_db"


def test_normalize_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        _normalize_recipe({"source": "bogus", "assembly": "hg38"})


def test_normalize_requires_source_field():
    with pytest.raises(ValueError, match="missing 'source' field"):
        _normalize_recipe({"assembly": "hg38"})


def test_load_yaml_rejects_unknown_version(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 99\ngenome: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported registry schema version"):
        _load_yaml(bad)


@pytest.mark.parametrize(
    ("recipe", "field"),
    [
        ({"source": "ucsc"}, "assembly"),
        ({"source": "ucsc-hub"}, "accession"),
        ({"source": "ncbi"}, "accession"),
        ({"source": "manual"}, "content"),
        ({"source": "local"}, "path"),
        ({"source": "s3"}, "name"),
    ],
)
def test_validate_recipe_missing_field(recipe: dict, field: str):
    with pytest.raises(ValueError, match=f"missing fields.*'{field}'"):
        _validate_recipe(recipe)


def test_validate_recipe_passes_for_complete_recipe():
    # Should not raise.
    _validate_recipe({"source": "ucsc", "assembly": "hg38"})
    _validate_recipe({"source": "local", "path": "/some/path"})
    _validate_recipe({"source": "s3", "name": "hg38"})
    _validate_recipe({"source": "manual", "content": "ACGT"})


def test_bundled_yaml_has_version_1():
    """All entries in the bundled YAML should be ucsc/source=ucsc with assembly set."""
    with open(_BUNDLED, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data["version"] == 1
    for name, entry in data["genome"].items():
        assert entry["source"] == "ucsc", f"{name} has non-ucsc source"
        assert "assembly" in entry, f"{name} missing assembly"
