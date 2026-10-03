"""Single source of the package version.

pyproject.toml reads it from here (hatchling), and it becomes the ``service.version`` on
everything the service emits, identifying the detection rules. Bump the minor version
whenever what gets detected changes.
"""

__version__ = "0.3.2"
