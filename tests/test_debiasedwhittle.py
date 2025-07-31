from debiased_spatial_whittle import __version__
import tomli


def test_version():
    """
    This test checks that the version declared in the pyproject.toml and in the __version__ variable of the package
    match.
    """
    with open("../pyproject.toml", "rb") as f:
        toml_parse = tomli.load(f)
    assert __version__ == toml_parse["tool"]["poetry"]["version"]
