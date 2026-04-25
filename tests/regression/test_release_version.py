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
    changelog_headings: tuple[str, ...] = ("0.2.0",),
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

    if changelog_headings:
        changelog = ["# Changelog", ""]
        for heading in changelog_headings:
            changelog.extend(
                [
                    f"## {heading}",
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
    ["0.2.0.dev1", "0.2.0rc1", "0.2.0"],
)
def test_validate_version_string_accepts_policy_forms(raw_version):
    parsed = validate_version_string(raw_version, "Project version")

    assert str(parsed) == raw_version


@pytest.mark.parametrize(
    "raw_version",
    [
        "0.2",
        "0.2.0.post1",
        "0.2.0+local",
        "0.2.0-rc.1",
        "0.2.0c1",
        "0.2.0.rc1",
        "0.2.0dev1",
        "0.2.0RC1",
    ],
)
def test_validate_version_string_rejects_disallowed_forms(raw_version):
    with pytest.raises(ReleaseVersionError):
        validate_version_string(raw_version, "Project version")


@pytest.mark.parametrize(
    ("version", "heading", "expected_kind"),
    [
        ("0.2.3.dev0", "Unreleased", "dev"),
        ("0.2.3.dev1", "Unreleased", "dev"),
        ("0.2.3rc1", "0.2.3rc1", "rc"),
        ("0.2.3", "0.2.3", "final"),
    ],
)
def test_source_mode_accepts_dev_rc_and_final_versions(
    tmp_path,
    version,
    heading,
    expected_kind,
):
    citation_version = "0.2.3" if expected_kind == "final" else "0.2.2"
    _write_release_files(
        tmp_path,
        version=version,
        citation_version=citation_version,
        changelog_headings=(heading,),
    )

    result = check_release_version(
        tmp_path,
        event_name="workflow_dispatch",
    )

    assert result.version == version
    assert result.tag == ""
    assert result.release_kind == expected_kind


def test_branch_push_with_dev_version_uses_source_mode(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=("Unreleased",),
    )

    result = check_release_version(
        tmp_path,
        event_name="push",
        ref_name="main",
    )

    assert result.version == "0.2.3.dev0"
    assert result.release_kind == "dev"
    assert result.release_mode is False
    assert result.is_prerelease is True


def test_source_mode_dev_requires_unreleased_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=("0.2.3",),
    )

    with pytest.raises(ReleaseVersionError, match="Unreleased"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


def test_source_mode_rc_accepts_exact_rc_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3rc1",
        citation_version="0.2.2",
        changelog_headings=("0.2.3rc1",),
    )

    result = check_release_version(
        tmp_path,
        event_name="workflow_dispatch",
    )

    assert result.release_kind == "rc"


def test_source_mode_rc_fails_without_exact_rc_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3rc1",
        citation_version="0.2.2",
        changelog_headings=("Unreleased",),
    )

    with pytest.raises(ReleaseVersionError, match="0.2.3rc1"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


@pytest.mark.parametrize("citation_version", [None, "0.2.2", "999.0.0"])
def test_source_mode_rc_does_not_require_matching_citation_version(
    tmp_path,
    citation_version,
):
    _write_release_files(
        tmp_path,
        version="0.2.3rc1",
        citation_version=citation_version,
        changelog_headings=("0.2.3rc1",),
    )

    result = check_release_version(
        tmp_path,
        event_name="workflow_dispatch",
    )

    assert result.version == "0.2.3rc1"
    assert result.release_kind == "rc"


def test_source_mode_final_requires_changelog_and_matching_citation(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3",
        citation_version="0.2.3",
        changelog_headings=("0.2.3",),
    )

    result = check_release_version(
        tmp_path,
        event_name="workflow_dispatch",
    )

    assert result.release_kind == "final"


def test_source_mode_final_fails_without_matching_citation_version(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3",
        citation_version="0.2.2",
        changelog_headings=("0.2.3",),
    )

    with pytest.raises(ReleaseVersionError, match="CITATION.cff version"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


def test_source_mode_final_requires_citation_file(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3",
        changelog_headings=("0.2.3",),
    )

    with pytest.raises(ReleaseVersionError, match="CITATION.cff must exist"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


def test_source_mode_final_fails_without_exact_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3",
        citation_version="0.2.3",
        changelog_headings=("Unreleased",),
    )

    with pytest.raises(ReleaseVersionError, match="0.2.3"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


def test_source_mode_requires_changelog_file(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=(),
    )

    with pytest.raises(ReleaseVersionError, match="CHANGELOG.md must exist"):
        check_release_version(
            tmp_path,
            event_name="workflow_dispatch",
        )


def test_tag_push_requires_matching_final_version_metadata(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_headings=("0.2.0",),
    )

    result = check_release_version(
        tmp_path,
        event_name="push",
        ref_name="v0.2.0",
    )

    assert result.version == "0.2.0"
    assert result.tag == "v0.2.0"
    assert result.release_kind == "final"
    assert result.is_prerelease is False


def test_tag_push_accepts_release_candidate_without_citation_match(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0rc1",
        citation_version="0.1.0",
        changelog_headings=("0.2.0rc1",),
    )

    result = check_release_version(
        tmp_path,
        event_name="push",
        ref_name="v0.2.0rc1",
    )

    assert result.version == "0.2.0rc1"
    assert result.tag == "v0.2.0rc1"
    assert result.release_kind == "rc"
    assert result.is_prerelease is True


def test_tag_push_rc_requires_exact_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0rc1",
        citation_version="0.1.0",
        changelog_headings=("Unreleased",),
    )

    with pytest.raises(ReleaseVersionError, match="0.2.0rc1"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.0rc1",
        )


def test_tag_push_rejects_development_versions_with_clear_error(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=("Unreleased",),
    )

    with pytest.raises(
        ReleaseVersionError,
        match="Development versions cannot be tagged",
    ):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.3.dev0",
        )


def test_tag_push_rejects_development_project_version_with_final_tag(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=("Unreleased",),
    )

    with pytest.raises(
        ReleaseVersionError,
        match="Development versions cannot be tagged.*bump",
    ):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.3",
        )


@pytest.mark.parametrize(
    "ref_name",
    ["v0.2.0c1", "v0.2.0RC1", "v0.2.0.rc1"],
)
def test_tag_push_rejects_non_policy_tag_versions(tmp_path, ref_name):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_headings=("0.2.0",),
    )

    with pytest.raises(ReleaseVersionError, match="Tag version"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name=ref_name,
        )


def test_tag_push_rejects_mismatched_project_and_tag_versions(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_headings=("0.2.0",),
    )

    with pytest.raises(ReleaseVersionError, match="Tag version"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.1",
        )


def test_tag_push_rejects_mismatched_final_citation_version(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.1",
        changelog_headings=("0.2.0",),
    )

    with pytest.raises(ReleaseVersionError, match="CITATION.cff version"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.0",
        )


def test_tag_push_requires_exact_final_changelog_heading(tmp_path):
    _write_release_files(
        tmp_path,
        version="0.2.0",
        citation_version="0.2.0",
        changelog_headings=("0.2.0 - 2026-04-18",),
    )

    with pytest.raises(ReleaseVersionError, match="0.2.0"):
        check_release_version(
            tmp_path,
            event_name="push",
            ref_name="v0.2.0",
        )


def test_manual_dry_run_emits_release_kind_output(tmp_path, monkeypatch):
    _write_release_files(
        tmp_path,
        version="0.2.3.dev0",
        citation_version="0.2.2",
        changelog_headings=("Unreleased",),
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
        "version=0.2.3.dev0",
        "tag=",
        "is_prerelease=true",
        "release_kind=dev",
    ]
