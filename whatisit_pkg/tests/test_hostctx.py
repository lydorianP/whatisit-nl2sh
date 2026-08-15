"""Tests for whatisit.hostctx: distro detection, package manager mapping,
and stable/volatile block formatting.

Every test isolates itself via monkeypatch -- none may read or write the real
~/.config or ~/.local/share.
"""
from __future__ import annotations

import time

import pytest

from whatisit import config as cfg_mod
from whatisit import hostctx


def _mock_os_release(monkeypatch, content: str | None):
    """Replace hostctx.Path so Path('/etc/os-release') returns fake data.

    The fake Path object also implements .name so _probe()'s shell resolution
    does not AttributeError.
    """
    class FakePath:
        def __init__(self, p):
            self._p = p
        def read_text(self):
            if self._p == "/etc/os-release":
                if content is None:
                    raise OSError("no such file")
                return content
            raise OSError(f"unexpected path: {self._p}")
        @property
        def name(self):
            return self._p.split("/")[-1]
    monkeypatch.setattr(hostctx, "Path", FakePath)


def _mock_platform(monkeypatch, system="Darwin", release="23.0.0", machine="x86_64"):
    fake = type("P", (), {
        "system": lambda self: system,
        "release": lambda self: release,
        "machine": lambda self: machine,
    })()
    monkeypatch.setattr(hostctx, "platform", fake)


# ---------------------------------------------------------------------------
# _distro_info
# ---------------------------------------------------------------------------

class TestDistroInfo:
    def test_reads_os_release_fields(self, monkeypatch):
        os_release = (
            'NAME="Ubuntu"\n'
            'VERSION="22.04.3 LTS (Jammy Jellyfish)"\n'
            'ID=ubuntu\n'
            'ID_LIKE=debian\n'
            'PRETTY_NAME="Ubuntu 22.04.3 LTS"\n'
            'VERSION_ID="22.04"\n'
        )
        _mock_os_release(monkeypatch, os_release)
        info = hostctx._distro_info()
        assert info["id"] == "ubuntu"
        assert info["name"] == "Ubuntu"
        assert info["version"] == "22.04.3 LTS (Jammy Jellyfish)"
        assert info["version_id"] == "22.04"
        assert info["id_like"] == ["debian"]

    def test_falls_back_to_platform_when_os_release_missing(self, monkeypatch):
        _mock_os_release(monkeypatch, None)
        _mock_platform(monkeypatch)
        info = hostctx._distro_info()
        assert info["id"] == ""
        assert info["name"] == "Darwin"
        assert info["version"] == ""
        assert info["version_id"] == ""
        assert info["id_like"] == []

    def test_pretty_name_is_fallback_for_name(self, monkeypatch):
        os_release = 'PRETTY_NAME="Fedora Linux 39"\nID=fedora\n'
        _mock_os_release(monkeypatch, os_release)
        info = hostctx._distro_info()
        assert info["name"] == "Fedora Linux 39"


# ---------------------------------------------------------------------------
# _distro
# ---------------------------------------------------------------------------

class TestDistro:
    def test_returns_pretty_name(self, monkeypatch):
        os_release = 'PRETTY_NAME="Archcraft"\nID=archcraft\n'
        _mock_os_release(monkeypatch, os_release)
        assert hostctx._distro() == "Archcraft"

    def test_falls_back_to_platform_system(self, monkeypatch):
        _mock_os_release(monkeypatch, None)
        _mock_platform(monkeypatch)
        assert hostctx._distro() == "Darwin"


# ---------------------------------------------------------------------------
# _canonical_pkg
# ---------------------------------------------------------------------------

class TestCanonicalPkg:
    @pytest.mark.parametrize("distro_id,id_like,expected", [
        ("ubuntu", ["debian"], "apt"),
        ("debian", [], "apt"),
        ("fedora", [], "dnf"),
        ("centos", [], "dnf"),
        ("arch", [], "pacman"),
        ("manjaro", [], "pacman"),
        ("archcraft", ["arch"], "pacman"),
        ("alpine", [], "apk"),
        ("void", [], "xbps"),
        ("opensuse-leap", [], "zypper"),
        ("opensuse-tumbleweed", [], "zypper"),
        ("nixos", [], "brew"),     # not in the map -> brew (universal fallback)
        ("", [], "brew"),          # empty id -> brew
    ])
    def test_maps_distros_to_canonical_pkg(self, distro_id, id_like, expected):
        assert hostctx._canonical_pkg(distro_id, id_like) == expected

    def test_id_like_fallback(self):
        """archcraft is not in the map, but arch is via ID_LIKE."""
        assert hostctx._canonical_pkg("archcraft", ["arch"]) == "pacman"

    def test_first_match_wins(self):
        """If a distro ID maps directly, ID_LIKE is not consulted."""
        assert hostctx._canonical_pkg("ubuntu", ["debian"]) == "apt"


# ---------------------------------------------------------------------------
# _probe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_includes_distro_version(self, monkeypatch):
        os_release = 'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["distro"] == "Ubuntu"
        assert facts["distro_version"] == "22.04"
        assert facts["pkg"] == "apt"  # declared by map; no binary found

    def test_falls_back_to_declared_pkg_when_no_binary_found(self, monkeypatch):
        os_release = 'NAME="Archcraft"\nID=archcraft\nID_LIKE=arch\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["pkg"] == "pacman"  # declared, even though binary not found

    def test_found_binary_wins_over_declared(self, monkeypatch):
        os_release = 'NAME="Ubuntu"\nID=ubuntu\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: "/usr/bin/apt" if x == "apt" else None)
        facts = hostctx._probe()
        assert facts["pkg"] == "apt"

    def test_unknown_pkg_falls_back_to_brew(self, monkeypatch):
        """Unmapped distros default to brew (the universal non-Linux answer)."""
        os_release = 'NAME="Unknown"\nID=unknown\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["pkg"] == "brew"

    def test_includes_present_and_missing_tools(self, monkeypatch):
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: "/usr/bin/git" if x == "git" else None)
        facts = hostctx._probe()
        assert "git" in facts["present"]
        assert "apt" in facts["missing"]


# ---------------------------------------------------------------------------
# stable_block
# ---------------------------------------------------------------------------

class TestStableBlock:
    def test_includes_version_when_present(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["git", "apt"],
            "missing": ["brew", "pacman"],
        }
        block = hostctx.stable_block(facts)
        assert "Ubuntu 22.04" in block

    def test_omits_version_when_empty(self):
        facts = {
            "distro": "Fedora",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "dnf",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        # The version should NOT appear as a number after the distro name.
        # "OS: Fedora (x86_64)" is fine; "OS: Fedora 39 (x86_64)" would include one.
        assert "Fedora 39" not in block
        assert "OS: Fedora (x86_64)" in block

    def test_apt_guidance_appears(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "apt install" in block
        assert "apt-get" in block

    def test_pacman_guidance_appears(self):
        facts = {
            "distro": "Archcraft",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "zsh",
            "pkg": "pacman",
            "present": ["git", "pacman"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "pacman -S" in block
        assert "pacman -Syy" in block

    def test_brew_guidance_appears(self):
        facts = {
            "distro": "macOS",
            "distro_version": "",
            "arch": "arm64",
            "shell": "zsh",
            "pkg": "brew",
            "present": ["git", "brew"],
            "missing": ["apt", "pacman"],
        }
        block = hostctx.stable_block(facts)
        assert "brew install" in block
        assert "apt-get" in block

    def test_netstat_to_ss_steer(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["ss", "git"],
            "missing": ["netstat", "lsof"],
        }
        block = hostctx.stable_block(facts)
        assert "ss -lptn" in block
        assert "netstat is unavailable" in block

    def test_lsof_missing_recommends_ss_or_fuser(self):
        facts = {
            "distro": "Archcraft",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "zsh",
            "pkg": "pacman",
            "present": ["ss", "git"],
            "missing": ["lsof"],
        }
        block = hostctx.stable_block(facts)
        assert "lsof is unavailable" in block
        assert "fuser" in block

    def test_alpine_guidance(self):
        facts = {
            "distro": "Alpine Linux",
            "distro_version": "3.19",
            "arch": "x86_64",
            "shell": "sh",
            "pkg": "apk",
            "present": ["git"],
            "missing": ["apt", "yum"],
        }
        block = hostctx.stable_block(facts)
        assert "apk add" in block
        assert "apt-get" in block
        assert "not available" in block

    def test_dnf_guidance(self):
        facts = {
            "distro": "Fedora Linux",
            "distro_version": "39",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "dnf",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "dnf install" in block
        assert "yum install" in block

    def test_block_is_capped_at_max_chars(self):
        # Feed an artificially long present list to test the cap.
        long_tools = ["tool" + str(i) for i in range(200)]
        facts = {
            "distro": "Test",
            "distro_version": "1.0",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": long_tools,
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert len(block) <= hostctx.MAX_STABLE_CHARS

    def test_delegates_to_stable_facts_when_no_facts_given(self, monkeypatch, tmp_path):
        """When called with no facts, stable_block uses the cached probe."""
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        block = hostctx.stable_block()
        assert facts["distro"] in block
        assert facts["pkg"] in block


# ---------------------------------------------------------------------------
# stable_facts caching
# ---------------------------------------------------------------------------

class TestStableFactsCache:
    def test_returns_cached_value_within_ttl(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        # Should return the same dict (not re-probe).
        cached = hostctx.stable_facts()
        assert cached["generated"] == facts["generated"]
        assert cached["distro"] == facts["distro"]

    def test_refresh_bypasses_cache(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts1 = hostctx.stable_facts(refresh=True)
        # Manually advance the timestamp to simulate time passing.
        cache_path = hostctx._cache_path()
        import json as _json
        raw = _json.loads(cache_path.read_text())
        raw["generated"] = time.time() - hostctx.CACHE_TTL - 1
        cache_path.write_text(_json.dumps(raw))
        facts2 = hostctx.stable_facts(refresh=True)
        assert facts2["generated"] > facts1["generated"]

    def test_corrupt_cache_file_is_rebuilt(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        cache_path = hostctx._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not json}")
        facts = hostctx.stable_facts(refresh=True)
        assert "distro" in facts

    def test_missing_cache_is_built(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        assert facts["generated"] > 0
        assert "distro" in facts
        assert "present" in facts
        assert "missing" in facts


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_returns_plain_prompt_when_disabled(self, monkeypatch):
        system, user = hostctx.build("install numpy", enabled=False)
        assert system == cfg_mod.SYSTEM_PROMPT
        assert user == "install numpy"

    def test_includes_system_prompt_plus_facts_when_enabled(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        system, user = hostctx.build("install numpy", enabled=True)
        assert system.startswith(cfg_mod.SYSTEM_PROMPT)
        assert "<host_environment>" in system
        assert "install numpy" in user

    def test_volatile_block_contains_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo.txt").write_text("hello")
        (tmp_path / "bar").mkdir()
        system, user = hostctx.build("list files", enabled=True)
        assert "foo.txt" in user
        assert "bar/" in user


class TestGrammar:
    """Tests for GBNF grammar generation."""

    def test_pacman_grammar_allows_pipes(self):
        g = hostctx.grammar_for_pkg("pacman")
        assert g is not None
        # other_cmd should be permissive (char+), allowing pipes
        assert "other_cmd" in g
        assert 'char' in g

    def test_pacman_grammar_allows_special_chars(self):
        g = hostctx.grammar_for_pkg("pacman")
        # The grammar should use [^\n] for other_cmd, allowing +, *, \, etc.
        assert '[^\\n]' in g or '[^\n]' in g

    def test_apt_grammar_allows_special_chars(self):
        g = hostctx.grammar_for_pkg("apt")
        assert g is not None
        assert 'char' in g

    def test_unknown_pkg_returns_none(self):
        assert hostctx.grammar_for_pkg("nonexistent") is None

    def test_all_known_pkgmgrs_have_grammar(self):
        for mgr in ("pacman", "apt", "dnf", "apk", "brew", "zypper", "xbps"):
            assert hostctx.grammar_for_pkg(mgr) is not None

    def test_grammar_constrains_install_verb(self):
        g = hostctx.grammar_for_pkg("pacman")
        # The install rule should mention pacman -S
        assert "pacman -S" in g

    def test_grammar_has_optional_trailing_newline(self):
        g = hostctx.grammar_for_pkg("pacman")
        # root should allow optional trailing newline: "\n"?
        assert '"\\n"?' in g


class TestPostprocess:
    """Tests for regex post-processing of wrong-distro commands."""

    def test_apt_to_pacman(self):
        assert hostctx.postprocess_command("apt-get install htop", "pacman") == "pacman -S htop"

    def test_sudo_apt_to_pacman(self):
        got = hostctx.postprocess_command("sudo apt-get install htop", "pacman")
        assert got == "sudo pacman -S htop"

    def test_dnf_to_pacman(self):
        assert hostctx.postprocess_command("dnf install htop", "pacman") == "pacman -S htop"

    def test_pacman_to_apt(self):
        assert hostctx.postprocess_command("pacman -S htop", "apt") == "apt install htop"

    def test_strips_deb_flags(self):
        got = hostctx.postprocess_command("apt-get install -y htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command(
            "apt-get install --no-install-recommends htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command("apt-get install -qq htop", "pacman")
        assert got == "pacman -S htop"

    def test_non_install_commands_unchanged(self):
        # Commands that don't match any rewrite rule pass through
        cmd = "find . -name '*.py' -exec wc -l {} ;"
        got = hostctx.postprocess_command(cmd, "pacman")
        assert got == cmd
        assert hostctx.postprocess_command("ps aux | grep nginx", "pacman") == "ps aux | grep nginx"
        assert hostctx.postprocess_command("ls -la > output.txt", "pacman") == "ls -la > output.txt"
        assert hostctx.postprocess_command("echo $HOME", "pacman") == "echo $HOME"

    def test_does_not_mangle_native_flags_on_non_apt_hosts(self):
        """Regression: deb-flag stripping must NOT run on commands that were
        not rewritten from an apt/legacy install syntax -- otherwise we corrupt
        native commands like `grep -qq` or `dnf install -y`."""
        # `-qq` is valid for grep; must survive untouched (no rewrite happened).
        cmd = "grep -qq pattern file"
        assert hostctx.postprocess_command(cmd, "pacman") == cmd
        assert hostctx.postprocess_command(cmd, "dnf") == cmd
        # `-y` is valid for dnf/yum/zypper (not only apt); native installs are
        # preserved because no rewrite took place.
        dnf_cmd = "dnf install -y htop"
        zypper_cmd = "zypper install -y htop"
        assert hostctx.postprocess_command(dnf_cmd, "dnf") == dnf_cmd
        assert hostctx.postprocess_command(zypper_cmd, "zypper") == zypper_cmd
        assert hostctx.postprocess_command("pacman -S htop", "pacman") == "pacman -S htop"

    def test_strips_deb_flags_only_after_a_rewrite(self):
        """After translating an apt command, apt-only flags are removed."""
        got = hostctx.postprocess_command("apt-get install -y htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command(
            "apt-get install --no-install-recommends htop", "pacman")
        assert got == "pacman -S htop"

    def test_multi_package_install(self):
        got = hostctx.postprocess_command("apt-get install htop nmon", "pacman")
        assert got == "pacman -S htop nmon"
