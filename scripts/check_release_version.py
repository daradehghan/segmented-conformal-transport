"""Validate package metadata for release and dry-run workflows."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tomllib

from packaging.version import InvalidVersion, Version
import yaml

COMMITTED_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:\.dev\d+|rc\d+)?$")
CHANGELOG_UNRELEASED_RE = re.compile(r"^##\s+Unreleased\s*$", re.MULTILINE)
ReleaseKind = Literal["dev", "rc", "final"]


class ReleaseVersionError(ValueError):
    """Raised when release metadata violates the repo policy."""


@dataclass(frozen=True)
class ReleaseCheckResult:
    """Hold the validated metadata needed by the release workflow."""

    version: str
    parsed_version: Version
    tag: str
    is_prerelease: bool
    release_kind: ReleaseKind
    release_mode: bool


def set_output(name: str, value: str) -> None:
    """Write a GitHub Actions step output when the runner requests it."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def _supported_forms_message() -> str:
    return "'X.Y.Z.devN', 'X.Y.ZrcN', or 'X.Y.Z'"


def _release_kind(parsed_version: Version) -> ReleaseKind:
    """Return the repository release kind for an already accepted version."""
    if parsed_version.is_devrelease:
        return "dev"
    if parsed_version.pre is not None:
        return "rc"
    return "final"


def validate_version_string(raw_version: str, label: str) -> Version:
    """Require PEP 440 parsing plus the repo's narrower committed forms."""
    try:
        parsed_version = Version(raw_version)
    except InvalidVersion as exc:
        raise ReleaseVersionError(
            f"{label} {raw_version!r} is not a valid Python package version."
        ) from exc

    if len(parsed_version.release) != 3:
        raise ReleaseVersionError(
            f"{label} {raw_version!r} must use exactly three release segments "
            "and one of the supported forms "
            f"{_supported_forms_message()}."
        )

    if parsed_version.post is not None or parsed_version.local is not None:
        raise ReleaseVersionError(
            f"{label} {raw_version!r} uses a post-release or local version, "
            "which is outside the repository release policy."
        )

    if not COMMITTED_VERSION_RE.fullmatch(raw_version) or str(parsed_version) != raw_version:
        raise ReleaseVersionError(
            f"{label} {raw_version!r} does not match the supported canonical "
            f"forms {_supported_forms_message()}."
        )

    return parsed_version


def read_project_version(pyproject_path: Path) -> tuple[str, Version]:
    """Read the canonical package version from pyproject metadata."""
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        raw_version = str(data["project"]["version"])
    except KeyError as exc:
        raise ReleaseVersionError(
            "pyproject.toml is missing [project].version."
        ) from exc
    return raw_version, validate_version_string(raw_version, "Project version")


def read_citation_version(citation_path: Path) -> str:
    """Read the citation version from CITATION.cff."""
    with citation_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "version" not in data:
        raise ReleaseVersionError("CITATION.cff is missing a top-level version field.")
    return str(data["version"])


def require_matching_citation_version(
    citation_path: Path,
    raw_project_version: str,
    project_version: Version,
) -> None:
    """Require CITATION.cff to agree with the package version."""
    citation_raw = read_citation_version(citation_path)
    try:
        citation_version = Version(citation_raw)
    except InvalidVersion as exc:
        raise ReleaseVersionError(
            f"CITATION.cff version {citation_raw!r} is not a valid Python package "
            "version."
        ) from exc
    if citation_raw != raw_project_version or citation_version != project_version:
        raise ReleaseVersionError(
            f"CITATION.cff version {citation_raw!r} does not match "
            f"pyproject.toml version {raw_project_version!r}."
        )


def changelog_has_exact_heading(changelog_path: Path, heading: str) -> bool:
    """Return True when CHANGELOG.md has exactly the requested h2 heading."""
    content = changelog_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    return pattern.search(content) is not None


def require_changelog_heading(changelog_path: Path, heading: str) -> None:
    """Require CHANGELOG.md to contain an exact h2 heading."""
    if not changelog_path.exists():
        raise ReleaseVersionError(
            "CHANGELOG.md must exist before release metadata validation."
        )
    if not changelog_has_exact_heading(changelog_path, heading):
        raise ReleaseVersionError(
            f"CHANGELOG.md does not contain an exact heading {heading!r}."
        )


def require_unreleased_changelog_heading(changelog_path: Path) -> None:
    """Require CHANGELOG.md to contain the canonical development heading."""
    if not changelog_path.exists():
        raise ReleaseVersionError(
            "CHANGELOG.md must exist before release metadata validation."
        )
    content = changelog_path.read_text(encoding="utf-8")
    if CHANGELOG_UNRELEASED_RE.search(content) is None:
        raise ReleaseVersionError(
            "CHANGELOG.md does not contain the required '## Unreleased' heading "
            "for development versions."
        )


def _validate_metadata_for_version_shape(
    *,
    release_kind: ReleaseKind,
    raw_project_version: str,
    parsed_project_version: Version,
    citation_path: Path,
    changelog_path: Path,
) -> None:
    """Apply metadata checks implied by the committed version shape."""
    if release_kind == "dev":
        require_unreleased_changelog_heading(changelog_path)
        return

    require_changelog_heading(changelog_path, raw_project_version)
    if release_kind == "final":
        if not citation_path.exists():
            raise ReleaseVersionError(
                "CITATION.cff must exist for final release validation."
            )
        require_matching_citation_version(
            citation_path,
            raw_project_version,
            parsed_project_version,
        )


def check_release_version(
    repo_root: Path,
    *,
    event_name: str | None = None,
    ref_name: str | None = None,
) -> ReleaseCheckResult:
    """Validate release metadata for tag pushes and manual dry runs."""
    pyproject_path = repo_root / "pyproject.toml"
    citation_path = repo_root / "CITATION.cff"
    changelog_path = repo_root / "CHANGELOG.md"

    raw_project_version, parsed_project_version = read_project_version(pyproject_path)
    release_kind = _release_kind(parsed_project_version)
    release_mode = event_name == "push" and bool(ref_name) and ref_name.startswith("v")

    tag = ""
    if release_mode:
        tag = str(ref_name)
        raw_tag_version = tag[1:]
        parsed_tag_version = validate_version_string(raw_tag_version, "Tag version")
        if release_kind == "dev":
            raise ReleaseVersionError(
                "Development versions cannot be tagged or published; bump to "
                "a release candidate or final version before pushing a release tag."
            )
        if parsed_tag_version != parsed_project_version:
            raise ReleaseVersionError(
                f"Tag version {raw_tag_version!r} does not match "
                f"pyproject.toml version {raw_project_version!r}."
            )

    _validate_metadata_for_version_shape(
        release_kind=release_kind,
        raw_project_version=raw_project_version,
        parsed_project_version=parsed_project_version,
        citation_path=citation_path,
        changelog_path=changelog_path,
    )

    return ReleaseCheckResult(
        version=raw_project_version,
        parsed_version=parsed_project_version,
        tag=tag,
        is_prerelease=parsed_project_version.is_prerelease,
        release_kind=release_kind,
        release_mode=release_mode,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for local and CI invocation."""
    parser = argparse.ArgumentParser(
        description="Validate package metadata for release workflows."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing pyproject.toml.",
    )
    parser.add_argument(
        "--event-name",
        type=str,
        default=None,
        help="Workflow event name. Defaults to GITHUB_EVENT_NAME.",
    )
    parser.add_argument(
        "--ref-name",
        type=str,
        default=None,
        help="Workflow ref name. Defaults to GITHUB_REF_NAME.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the release guard and emit step outputs for GitHub Actions."""
    args = parse_args(argv)
    event_name = args.event_name or os.environ.get("GITHUB_EVENT_NAME")
    ref_name = args.ref_name or os.environ.get("GITHUB_REF_NAME")
    result = check_release_version(
        args.repo_root,
        event_name=event_name,
        ref_name=ref_name,
    )

    set_output("version", result.version)
    set_output("tag", result.tag)
    # Kept for external consumers; release workflow routing must use release_kind.
    set_output("is_prerelease", str(result.is_prerelease).lower())
    set_output("release_kind", result.release_kind)

    if result.release_mode:
        print(
            f"Validated release tag {result.tag!r} against package version "
            f"{result.version!r} ({result.release_kind})."
        )
    else:
        print(
            f"Validated package version {result.version!r} "
            f"({result.release_kind}) for source mode."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
