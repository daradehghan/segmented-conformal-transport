"""Validate package metadata for release and dry-run workflows."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
import tomllib

from packaging.version import InvalidVersion, Version
import yaml

STRICT_RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(rc\d+)?$")


class ReleaseVersionError(ValueError):
    """Raised when release metadata violates the repo policy."""


@dataclass(frozen=True)
class ReleaseCheckResult:
    """Hold the validated metadata needed by the release workflow."""

    version: str
    parsed_version: Version
    tag: str
    is_prerelease: bool
    release_mode: bool


def set_output(name: str, value: str) -> None:
    """Write a GitHub Actions step output when the runner requests it."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def validate_version_string(raw_version: str, label: str) -> Version:
    """Require the repo's raw version policy and packaging semantics."""
    if not STRICT_RELEASE_VERSION_RE.fullmatch(raw_version):
        raise ReleaseVersionError(
            f"{label} {raw_version!r} does not match the supported release forms "
            "'X.Y.Z' or 'X.Y.ZrcN'."
        )
    try:
        return Version(raw_version)
    except InvalidVersion as exc:
        raise ReleaseVersionError(
            f"{label} {raw_version!r} is not a valid Python package version."
        ) from exc


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
    if citation_version != project_version:
        raise ReleaseVersionError(
            f"CITATION.cff version {citation_raw!r} does not match "
            f"pyproject.toml version {raw_project_version!r}."
        )


def changelog_has_entry(changelog_path: Path, version: str) -> bool:
    """Return True when CHANGELOG.md has a heading for the given version."""
    content = changelog_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^##\s+{re.escape(version)}\b", re.MULTILINE)
    return pattern.search(content) is not None


def require_changelog_entry(changelog_path: Path, version: str) -> None:
    """Require CHANGELOG.md to contain a release heading for the version."""
    if not changelog_path.exists():
        raise ReleaseVersionError(
            "CHANGELOG.md must exist before automated public releases."
        )
    if not changelog_has_entry(changelog_path, version):
        raise ReleaseVersionError(
            f"CHANGELOG.md does not contain an entry for version {version!r}."
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
    release_mode = event_name == "push" and bool(ref_name) and ref_name.startswith("v")

    tag = ""
    if release_mode:
        tag = str(ref_name)
        if not tag.startswith("v"):
            raise ReleaseVersionError(
                f"Release tag {tag!r} must start with the 'v' prefix."
            )
        raw_tag_version = tag[1:]
        parsed_tag_version = validate_version_string(raw_tag_version, "Tag version")
        if parsed_tag_version != parsed_project_version:
            raise ReleaseVersionError(
                f"Tag version {raw_tag_version!r} does not match "
                f"pyproject.toml version {raw_project_version!r}."
            )

    if citation_path.exists():
        require_matching_citation_version(
            citation_path,
            raw_project_version,
            parsed_project_version,
        )
    elif release_mode:
        raise ReleaseVersionError(
            "CITATION.cff must exist for tag-based release validation."
        )

    if release_mode:
        require_changelog_entry(changelog_path, raw_project_version)
    elif changelog_path.exists() and not changelog_has_entry(
        changelog_path, raw_project_version
    ):
        print(
            f"CHANGELOG.md does not yet contain an entry for version "
            f"{raw_project_version!r}.",
        )

    return ReleaseCheckResult(
        version=raw_project_version,
        parsed_version=parsed_project_version,
        tag=tag,
        is_prerelease=parsed_project_version.is_prerelease,
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
    set_output("is_prerelease", str(result.is_prerelease).lower())

    if result.release_mode:
        print(
            f"Validated release tag {result.tag!r} against package version "
            f"{result.version!r}."
        )
    else:
        print(f"Validated package version {result.version!r} for dry-run mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
