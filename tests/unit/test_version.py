import importlib.metadata
import re
from pathlib import Path

from opentelemetry.sdk.resources import SERVICE_VERSION

import llm_eval_otel
from llm_eval_otel.emit.sdk import resource
from llm_eval_otel.version import __version__

VERSION_FILE = Path(llm_eval_otel.__file__).with_name("version.py")


def test_version_line_is_what_the_scripts_read() -> None:
    # release.yml and scripts/push-nexus.sh read this exact line with sed, and the
    # release tag must be plain SemVer for docker/metadata-action.
    lines = re.findall(r'^__version__ = "(.*)"$', VERSION_FILE.read_text(), re.MULTILINE)
    assert lines == [__version__]
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_package_metadata_uses_version_py() -> None:
    assert importlib.metadata.version("llm-eval-otel") == __version__
    assert llm_eval_otel.__version__ == __version__


def test_resource_carries_the_version() -> None:
    assert resource().attributes[SERVICE_VERSION] == __version__
