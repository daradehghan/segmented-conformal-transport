import textwrap

import pytest

from scripts.check_release_version import (
    ReleaseVersionError,
    check_release_version,
    main,
    validate_version_string,
)


def _write_release_files(
    repo_root,
    *,
    version: str = "0.2.0",
    citation_version: str | None = None,
    changelog_versions: tuple[str, ...] = ("0.2.0",),
) -> None:
    repo_root.joinpath("pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "tsconformal"
            version = "{version}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    if citation_version is not None:
        repo_root.joinpath("CITATION.cff").write_text(
            textwrap.dedent(
                f"""
                cff-version: 1.2.0
                title: tsconformal
                version: {citation_version}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    if changelog_versions:
        changelog = ["# Changelog", ""]
        for release_version in changelog_versions:
            changelog.extend(
                [
                    f"## {release_version} - 2026-04-18",
                    "",
                    "### Changed",
                    "",
                    "- Metadata validation tightened.",
                    "",
                ]
            )
        repo_root.joinpath("CHANGELOG.md").write_text(
            "\n".join(changelog),
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    "raw_version",
    ["0.2", "0.2.0.dev1", "0.2.0.post1", "0.2.0+local", "0.2.0-rc.1"],
)
def test_validate_version_string_rejects_disallowed_forms(raw_version):
    with pytest.raises(ReleaseVersionError):
        validate_version_string(raw_version, "Project version")


def test_tag_push_requires_matching_version_metadata(tmp_path):
    _write_release_files(tmp_path, version="0.2.0", citation_version="0.2.0")

    result = check_release_version(
        tmp_path,
        event_name="push",
        ref_name="v0.2.0",
    )

    assert result.version == "0.2.0"
    assert result.tag == "v0.2.0"
    assert result.is_prerelease is False


def test_tag_push_marks_release_candidates_as_prereleases(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0rc1",
        citation_version="0.2.0rc1",
        changelog_versions=("0.2.0rc1",),
    )

    result = check_release_version(
        tmp_path,
        event_name="push",
        ref_name="v0.2.0rc1",
    )

    assert result.version == "0.2.0rc1"
    assert result.tag == "v0.2.0rc1"
    assert result.is_prerelease is True


def test_tag_push_rejects_mismatched_citation_version(tmp_path):
    _write_release_files(tmp_path, version="0.2.0", citation_version="0.2.1")

    with pytest.raises(ReleaseVersionError, match="CITATION.cff version"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.0",
        )


def test_tag_push_requires_matching_changelog_entry(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_versions=("0.1.0",),
    )

    with pytest.raises(ReleaseVersionError, match="CHANGELOG.md does not contain"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.0",
        )


def test_manual_dry_run_does_not_require_a_tag(tmp_path, monkeypatch):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_versions=(),
    )
    github_output = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--event-name",
            "workflow_dispatch",
        ]
    )

    assert exit_code == 0
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "version=0.2.0",
        "tag=",
        "is_prerelease=false",
    ]
