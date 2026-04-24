"""Tests for persistent directive configuration."""

from prompterator.config.schema import Config


def test_directives_default_is_none():
    """Directives config defaults to empty — no persistent directive."""
    config = Config()
    assert config.directives.default is None
    assert config.directives.issues is None
    assert config.directives.evals is None
    assert config.directives.improve is None


def test_resolve_directive_empty_returns_none():
    config = Config()
    assert config.resolve_directive("issues") is None
    assert config.resolve_directive("evals") is None
    assert config.resolve_directive("improve") is None


def test_resolve_directive_falls_back_to_default():
    config = Config(directives={"default": "be careful"})
    assert config.resolve_directive("issues") == "be careful"
    assert config.resolve_directive("evals") == "be careful"
    assert config.resolve_directive("improve") == "be careful"


def test_resolve_directive_command_specific_overrides_default():
    config = Config(directives={
        "default": "generic",
        "improve": "preserve YAML frontmatter",
    })
    assert config.resolve_directive("improve") == "preserve YAML frontmatter"
    # other commands still get the default
    assert config.resolve_directive("issues") == "generic"
    assert config.resolve_directive("evals") == "generic"


def test_directives_absent_from_yaml_when_empty():
    config = Config()
    d = config.to_yaml_dict()
    assert "directives" not in d


def test_directives_serialized_only_when_set():
    config = Config(directives={
        "default": "always do X",
        "improve": "and especially do Y when improving",
    })
    d = config.to_yaml_dict()
    assert d["directives"] == {
        "default": "always do X",
        "improve": "and especially do Y when improving",
    }
    # unset command-specific fields are not serialized
    assert "issues" not in d["directives"]
    assert "evals" not in d["directives"]
