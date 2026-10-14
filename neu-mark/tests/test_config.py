"""Site defaults: a URL builder, never a silent fallback."""

import pytest

from neu_mark import config

TOML = """
[dvid]
server = "dvid.example.org"
uuid = "93fdbc:main"
locked = true

[instances]
synapses = "synapses"
bodies = "labels_annotations"
counts = "synapses_labelsz"

[rule_sets]
wasp = "rules/wasp_rules.py"
absolute = "/opt/shared/rules.py"
"""


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "neu-mark.toml"
    path.write_text(TOML)
    monkeypatch.setenv(config.ENV_VAR, str(path))
    return config.load()


@pytest.mark.parametrize("args, kwargs, expected", [
    (("synapses",), {}, "dvid://dvid.example.org/93fdbc:main/synapses"),
    (("counts",), {}, "dvid://dvid.example.org/93fdbc:main/synapses_labelsz"),
    (("labels_todo",), {}, "dvid://dvid.example.org/93fdbc:main/labels_todo"),  # literal name
    (("synapses",), {"uuid": "846e3a"}, "dvid://dvid.example.org/846e3a/synapses"),
])
def test_it_builds_a_complete_url(cfg, args, kwargs, expected):
    """Complete, so it can be printed and put in a saved command."""
    assert cfg.url(*args, **kwargs) == expected
    assert cfg.locked is True


def test_resolve_only_touches_the_at_prefix(cfg):
    """The prefix is required so a saved command visibly depends on a config."""
    assert cfg.resolve("@synapses").endswith("/synapses")
    assert cfg.resolve("dvid://other/abc/labels") == "dvid://other/abc/labels"
    assert cfg.resolve("/some/path") == "/some/path"


def test_an_unknown_reference_lists_what_is_configured(cfg):
    with pytest.raises(ValueError, match="Configured names: bodies, counts, synapses"):
        cfg.resolve("@nope")
    with pytest.raises(ValueError, match="names nothing after"):
        cfg.resolve("@")


def test_no_config_loads_empty_rather_than_raising(tmp_path, monkeypatch):
    """Importing must never depend on a file existing. The `chdir` matters: without it the
    upward walk finds the developer's own config, and nothing machine-wide is consulted."""
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert config.find() is None, "tmp_path is not isolated; the walk found a real config"
    assert config.find(tmp_path) is None

    empty = config.load()
    assert empty.path is None and empty.server is None
    assert empty.instances == {} and empty.roi_sets == {} and empty.rule_sets == {}


def test_an_empty_config_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="sets no dvid.server"):
        config.Config({}, None).url("synapses")
    with pytest.raises(ValueError, match="in the working directory and every parent"):
        config.Config({}, None).resolve("@synapses")

    path = tmp_path / "c.toml"
    path.write_text('[dvid]\nserver = "host"\n')
    with pytest.raises(ValueError, match="sets no dvid.uuid"):
        config.load(str(path)).url("synapses")


def test_an_explicit_env_path_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    """Falling through would silently use a different file than the one named."""
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match="is not a file"):
        config.load()


def test_load_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(TOML)
    assert config.load(str(path)).uuid == "93fdbc:main"


# --------------------------------------------------------------------------- #
# finding it: an upward walk
# --------------------------------------------------------------------------- #
def test_it_is_found_by_walking_up_from_a_subdirectory(tmp_path, monkeypatch):
    """Notebooks sit a few directories below where the file lives."""
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / config.FILENAME).write_text(TOML)
    deep = tmp_path / "neu-mark" / "notebooks"
    deep.mkdir(parents=True)
    assert config.find(deep) == str(tmp_path / config.FILENAME)
    monkeypatch.chdir(deep)
    assert config.find() == str(tmp_path / config.FILENAME)
    assert config.load().server == "dvid.example.org"


def test_it_is_found_in_a_configs_subdirectory_on_the_way_up(tmp_path, monkeypatch):
    """Rules modules named in it resolve against the config's own directory."""
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    (tmp_path / config.SUBDIR).mkdir()
    path = tmp_path / config.SUBDIR / config.FILENAME
    path.write_text(TOML)
    deep = tmp_path / "scripts" / "eval"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert config.find() == str(path)
    assert config.load().rule_set("wasp") == str(tmp_path / config.SUBDIR / "rules" / "wasp_rules.py")


@pytest.mark.parametrize("layout, start, expected", [
    ({"": "outer", "inner": "nearer"}, "inner", "nearer"),          # the nearest wins
    ({config.SUBDIR: "sub", "": "bare"}, "", "bare"),               # bare beats configs/
])
def test_which_config_wins(tmp_path, monkeypatch, layout, start, expected):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    for where, server in layout.items():
        (tmp_path / where).mkdir(exist_ok=True)
        (tmp_path / where / config.FILENAME).write_text(f'[dvid]\nserver = "{server}"\n')
    monkeypatch.chdir(tmp_path / start)
    assert config.load().server == expected


def test_a_named_rules_module_resolves_against_the_config_not_the_cwd(cfg, tmp_path,
                                                                     monkeypatch):
    """So a notebook and a shell at the workspace root load the same module."""
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert cfg.rule_set("wasp") == str(tmp_path / "rules" / "wasp_rules.py")
    assert cfg.rule_set("absolute") == "/opt/shared/rules.py"
    with pytest.raises(ValueError, match="absolute, wasp"):
        cfg.rule_set("nope")
