r"""Tell the model about the machine it is running on.

Why this exists, and why it is shaped this way.

The single most common failure in real use was the model emitting a placeholder
(`cp -r /path/to/dir /tmp`) or naming a tool the host does not have
(`netstat -tuln | grep 5000` on a box with no netstat). Both are the same defect:
the model is guessing at facts it could simply be told.

Measured on the target hardware: putting a host-facts block in the SYSTEM prompt
is nearly free, because llama-server prefix-caches it. A 685-token block cost
11.1 s on the first query and then 1.25 / 0.54 / 1.92 s -- i.e. baseline. Paid
once per session. In the same test the block changed
`netstat -tuln | grep 5000` into `ss -lptn | grep 5000`, fixing a real observed
failure, because the block said netstat was absent and ss shows pids.

Two rules follow from that measurement:

STABLE facts (distro, package manager, which tools exist, shell) go in the
system prompt so the cached prefix stays byte-identical across queries.

VOLATILE facts (cwd, directory listing, git branch) must NOT go there -- they
change per query and would invalidate the cache every time, turning a once-per-
session cost into a per-query one. They are appended to the user turn instead.

Budget: the stable block is capped (see MAX_STABLE_CHARS). The 685-token block
used in the experiment was deliberately inflated to measure the worst case; a
real one is ~150 tokens, about 2.4 s of one-time prefill.

STATUS: DISABLED BY DEFAULT, on measurement.

I originally ranked this the highest-value change on the strength of 4 hand-run
examples. A proper 295-task run reversed that, and the anecdote was simply too
small to rank on:

    no context   0.849 mean / 54.2% pass
    context      0.815 mean / 45.1% pass    d=-0.034, McNemar p=0.0004

Two implementation bugs accounted for part of it (a false working-directory claim
and a key=value format the model read as shell variables); fixing both recovered
39.0% -> 45.1% but did not close the gap. Ruled out as the cause: only 3 of 41
regressions involve a tool the context declared missing.

What actually happens is that the extra tokens make a 1.5B model produce more
elaborate answers, and elaboration breaks simple correct ones:
    `sha512sum f`            -> `echo -n "f" | sha512sum`   (hashes the NAME)
    `echo 'hello' > world.txt` -> `touch /testbed/world.txt`  (drops the content)
    `find .. | xargs wc -l`  -> `find .. -exec wc -l {} \;`  (working idiom traded away)

But the benchmark is close to blind to the BENEFIT this is for. Measured on the
task text: 73% of ALFA tasks already name a concrete target path, so context is
pure noise there, and the underspecified 27% are trivia (`ls`, `pwd`, `date`)
that need no context either. Of real typed queries, 79% are underspecified --
the opposite distribution.

So ALFA measures this feature's cost in full and its benefit barely. The measured
harm is real and sufficient reason not to enable it by default; it is NOT
sufficient reason to conclude the idea is wrong. Revisit only against an eval
built from underspecified requests.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config as cfg_mod

CACHE_TTL = 24 * 3600      # installed tools change rarely
MAX_STABLE_CHARS = 900     # ~200 tokens; keeps first-query prefill near 2-3 s
MAX_ENTRIES = 12           # directory listing truncation

# Tools worth telling the model about, chosen from observed failures rather than
# from a generic "useful commands" list: each one is either a tool the model
# reached for and the host lacked, or the correct modern replacement for one.
PROBE_TOOLS = [
    "ss", "netstat", "lsof", "ip", "ifconfig",       # the port-lookup failures
    "docker", "podman", "kubectl",
    "git", "rg", "fd", "jq", "tree", "ncdu",
    "zip", "unzip", "tar", "xz", "rsync", "curl", "wget",
    "python3", "node", "shellcheck",
    "squeue", "sbatch",                              # HPC: scheduler present?
    "systemctl", "brew", "apt", "dnf", "yum", "pacman", "apk",
]

PKG_MANAGERS = [("apt", "apt"), ("dnf", "dnf"), ("yum", "yum"),
                ("pacman", "pacman"), ("apk", "apk"), ("brew", "brew"),
                ("zypper", "zypper")]


def _distro_info() -> dict:
    """Return structured distro metadata from /etc/os-release.

    Falls back to platform.system() / platform.release() when the file is
    absent (e.g. macOS, WSL without os-release, minimal containers).
    """
    try:
        data = dict(
            line.split("=", 1)
            for line in Path("/etc/os-release").read_text().splitlines()
            if "=" in line
        )
        return {
            "id": data.get("ID", "").strip('"').lower(),
            "name": data.get("NAME", data.get("PRETTY_NAME", "")).strip('"'),
            "version": data.get("VERSION", "").strip('"'),
            "version_id": data.get("VERSION_ID", "").strip('"'),
            "id_like": [v.strip('"').lower() for v in data.get("ID_LIKE", "").split()],
        }
    except OSError:
        return {
            "id": "",
            "name": platform.system(),
            "version": "",
            "version_id": "",
            "id_like": [],
        }


def _distro() -> str:
    info = _distro_info()
    if info["name"]:
        return info["name"]
    return platform.system()


# Canonical package manager for each well-known distro ID. Checked in order; the
# first match wins. This is a DECLARATIVE hint, not a runtime check -- the real
# presence check is done by PKG_MANAGERS / shutil.which below. It lets the
# stable block name the RIGHT package manager even on distros (e.g. Archcraft,
# NixOS) where the binary name differs from the distro ID.
DISTRO_PKG_MAP: dict[str, str] = {
    "ubuntu": "apt",
    "debian": "apt",
    "linuxmint": "apt",
    "pop": "apt",
    "zorin": "apt",
    "elementary": "apt",
    "mx": "apt",
    "raspbian": "apt",
    "kali": "apt",
    "fedora": "dnf",
    "centos": "dnf",
    "rhel": "dnf",
    "rocky": "dnf",
    "almalinux": "dnf",
    "amazon": "dnf",
    "opensuse-leap": "zypper",
    "opensuse-tumbleweed": "zypper",
    "opensuse": "zypper",
    "sles": "zypper",
    "arch": "pacman",
    "manjaro": "pacman",
    "endeavouros": "pacman",
    "cachyos": "pacman",
    "garuda": "pacman",
    "artix": "pacman",
    "alpine": "apk",
    "void": "xbps",
}


def _canonical_pkg(distro_id: str, id_like: list[str]) -> str:
    """Map a distro ID (and its ID_LIKE family) to the canonical package manager.

    This is a DECLARATIVE map: the actual binary-presence check is done by the
    PKG_MANAGERS loop in _probe(). Here we only pick the name the model should
    use when it talks about this machine's package manager. That matters for
    distros like Archcraft (ID=archcraft, ID_LIKE=arch) where the binary is
    `pacman` but the ID is not in DISTRO_PKG_MAP by itself.
    """
    checked: set[str] = set()
    for candidate in (distro_id,) + tuple(id_like):
        if candidate in checked:
            continue
        checked.add(candidate)
        if pkg := DISTRO_PKG_MAP.get(candidate):
            return pkg
    # macOS and anything without os-release fields: Homebrew is the universal
    # answer the model already knows.
    return "brew"


def _cache_path() -> Path:
    return cfg_mod.data_dir() / "hostctx.json"


def _probe() -> dict:
    info = _distro_info()
    present = [t for t in PROBE_TOOLS if shutil.which(t)]
    missing = [t for t in PROBE_TOOLS if t not in present]
    # The canonical pkg name from the distro map; fall back to the binary we
    # actually found on PATH. This means the block names the right tool even
    # when the distro ID is a custom spin (Archcraft, NixOS, etc.) whose
    # binary is the upstream Arch one.
    decl_pkg = _canonical_pkg(info["id"], info["id_like"])
    found_pkg = next((name for bin_, name in PKG_MANAGERS if shutil.which(bin_)), "unknown")
    pkg = found_pkg if found_pkg != "unknown" else decl_pkg
    return {
        "generated": time.time(),
        "distro": info["name"] or platform.system(),
        "distro_id": info["id"],
        "distro_version": info["version_id"] or info["version"],
        "kernel": platform.release(),
        "arch": platform.machine(),
        "shell": Path(os.environ.get("SHELL", "/bin/sh")).name,
        "pkg": pkg,
        "present": present,
        "missing": missing,
    }


def stable_facts(refresh: bool = False) -> dict:
    """Probe the host, cached to disk. Cheap after the first call."""
    p = _cache_path()
    if not refresh and p.exists():
        try:
            d = json.loads(p.read_text())
            if time.time() - d.get("generated", 0) < CACHE_TTL:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    d = _probe()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2))
    except OSError:
        pass
    return d


# Few-shot examples keyed by package manager. These are injected into the
# system prompt to override the model's training bias with a concrete example.
# The example mirrors the exact command style the model should emit.
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "apt": "User: install htop\nAssistant: sudo apt install htop",
    "dnf": "User: install htop\nAssistant: sudo dnf install htop",
    "pacman": "User: install htop\nAssistant: sudo pacman -S htop",
    "apk": "User: install htop\nAssistant: apk add htop",
    "brew": "User: install htop\nAssistant: brew install htop",
    "zypper": "User: install htop\nAssistant: sudo zypper install htop",
    "xbps": "User: install htop\nAssistant: xbps-install htop",
}


def stable_block(facts: dict | None = None) -> str:
    """The part that goes in the system prompt. Must be stable across queries."""
    f = facts or stable_facts()
    # Include version when present so the model can give version-specific
    # commands (e.g. apt vs. apt-get, dnf vs. yum, brew vs. port).
    version_tag = f" {f['distro_version']}" if f.get("distro_version") else ""
    lines = [
        "<host_environment>",
        f"OS: {f['distro']}{version_tag} ({f['arch']})",
        f"Shell: {f['shell']}",
        f"Package manager: {f['pkg']}",
        f"Available tools: {' '.join(f['present'])}",
        "</host_environment>",
    ]
    # NOTE: We deliberately omit a "Banned tools" line. Small models (<3B)
    # prime on negative constraints — mentioning `apt` in the prompt increases
    # the probability the model outputs `apt-get`, even inside a "banned" list.
    # The <example> block and <constraint> tags provide positive guidance
    # without activating the wrong concepts.
    # Steer to the modern tool when the legacy one is genuinely unavailable.
    if "ss" in f["present"] and "netstat" in f["missing"]:
        lines.append("<constraint>")
        lines.append("For listening ports use `ss -lptn` (shows the owning pid); "
                     "netstat is unavailable.")
        lines.append("</constraint>")
    if "lsof" in f["missing"] and "ss" in f["present"]:
        lines.append("<constraint>")
        lines.append("lsof is unavailable; use `ss -lptn` or `fuser` instead.")
        lines.append("</constraint>")
    # Distro-specific package manager guidance: some distros ship legacy names
    # alongside modern ones, and the model should use the canonical one.
    if f["pkg"] == "apt":
        lines.append("<constraint>")
        lines.append("Install packages with `apt install <pkg>` (not `apt-get install`).")
        lines.append("</constraint>")
    elif f["pkg"] == "dnf":
        lines.append("<constraint>")
        lines.append("Install packages with `dnf install <pkg>` (not `yum install`).")
        lines.append("</constraint>")
    elif f["pkg"] == "pacman":
        lines.append("<constraint>")
        lines.append("Install packages with `pacman -S <pkg>`; "
                      "use `pacman -Syy` to force a full refresh.")
        lines.append("</constraint>")
    elif f["pkg"] == "apk":
        lines.append("<constraint>")
        lines.append("Install packages with `apk add <pkg>`. "
                      "On Alpine, `apt-get` is not available.")
        lines.append("</constraint>")
    elif f["pkg"] == "brew":
        lines.append("<constraint>")
        lines.append("Install packages with `brew install <pkg>`. "
                      "`apt-get` and `yum` are not available on macOS.")
        lines.append("</constraint>")
    # Few-shot example: inject a concrete usage example for this distro's pkg mgr.
    if example := FEW_SHOT_EXAMPLES.get(f["pkg"]):
        lines.append("<example>")
        lines.append(example)
        lines.append("</example>")
    block = "\n".join(lines)
    return block[:MAX_STABLE_CHARS]


def volatile_block(cwd: Path | None = None) -> str:
    """Per-query facts. Deliberately NOT in the system prompt -- see module docstring."""
    cwd = Path(cwd or Path.cwd())
    # Prose labels, NOT `key=value`. Measured: a `cwd=/testbed` line made the
    # model treat the key as a shell variable and emit `mkdir -p $cwd/test_dir`
    # and `for i in $(echo $cwd_entries ...)`. The context format itself was
    # teaching it to reference variables that do not exist.
    lines = [f"Working directory is {cwd}"]
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in cwd.iterdir()
                         if not p.name.startswith("."))
        shown = entries[:MAX_ENTRIES]
        more = f" (+{len(entries)-len(shown)} more)" if len(entries) > len(shown) else ""
        lines.append(f"It contains: {' '.join(shown)}{more}" if shown else "It is empty.")
    except OSError:
        pass
    if git := _git_state(cwd):
        lines.append(git)
    return "\n".join(lines)


def _git_state(cwd: Path) -> str:
    """Branch and dirtiness, or '' if not a repo. Short timeout: a slow NFS repo
    must never delay a command suggestion."""
    if not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        if r.returncode != 0:
            return ""
        branch = r.stdout.strip()
        s = subprocess.run(["git", "status", "--porcelain"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        dirty = "dirty" if s.stdout.strip() else "clean"
        return f"It is a git repo on branch {branch}, working tree {dirty}."
    except (subprocess.TimeoutExpired, OSError):
        return ""


# Regex substitutions that translate a model's output from one distro's syntax
# to another. Applied as a duct-tape fallback when the 1.5B model gets the
# intent right but the syntax wrong (e.g. `apt-get install` on Arch).
# Each pattern is deliberately broad to catch variants the model might emit.
# Debian/Ubuntu flags that have no equivalent on most other distros and should
# be stripped during translation. `-y` / `-qq` are apt-specific;
# `--no-install-recommends` is also apt-specific (pacman has no equivalent).
# NOTE: this is only applied below when an apt command was actually rewritten
# to the host's syntax -- applied unconditionally it would mangle native
# commands that happen to contain e.g. `grep -qq` or `dnf install -y`.
_DEB_FLAGS = re.compile(r"(-y\s+|-qq\s+|--no-install-recommends\s+)+")

PKG_MGR_REWRITE: list[tuple[str, str, str]] = [
    # pacman host: translate Debian/Ubuntu/Fedora syntax -> Arch
    ("pacman",
     re.compile(r"\bapt(-get)?\s+install\b"), "pacman -S"),
    ("pacman",
     re.compile(r"\bsudo\s+apt(-get)?\s+install\b"), "sudo pacman -S"),
    ("pacman",
     re.compile(r"\bdnf\s+install\b"), "pacman -S"),
    ("pacman",
     re.compile(r"\bsudo\s+dnf\s+install\b"), "sudo pacman -S"),
    # apt host: translate Arch syntax -> Debian
    ("apt",
     re.compile(r"\bpacman\s+-S\b"), "apt install"),
    ("apt",
     re.compile(r"\bsudo\s+pacman\s+-S\b"), "sudo apt install"),
    # dnf host: translate Arch or Debian syntax -> Fedora
    ("dnf",
     re.compile(r"\bpacman\s+-S\b"), "dnf install"),
    ("dnf",
     re.compile(r"\bsudo\s+pacman\s+-S\b"), "sudo dnf install"),
    ("dnf",
     re.compile(r"\bapt(-get)?\s+install\b"), "dnf install"),
    ("dnf",
     re.compile(r"\bsudo\s+apt(-get)?\s+install\b"), "sudo dnf install"),
    # apk host: translate everything else -> Alpine
    ("apk",
     re.compile(r"\bapt(-get)?\s+install\b"), "apk add"),
    ("apk",
     re.compile(r"\bsudo\s+apt(-get)?\s+install\b"), "apk add"),
    ("apk",
     re.compile(r"\bpacman\s+-S\b"), "apk add"),
    ("apk",
     re.compile(r"\bsudo\s+pacman\s+-S\b"), "apk add"),
    ("apk",
     re.compile(r"\bdnf\s+install\b"), "apk add"),
    ("apk",
     re.compile(r"\bsudo\s+dnf\s+install\b"), "apk add"),
    # brew host: translate Linux syntax -> macOS
    ("brew",
     re.compile(r"\bapt(-get)?\s+install\b"), "brew install"),
    ("brew",
     re.compile(r"\bsudo\s+apt(-get)?\s+install\b"), "brew install"),
    ("brew",
     re.compile(r"\bpacman\s+-S\b"), "brew install"),
    ("brew",
     re.compile(r"\bsudo\s+pacman\s+-S\b"), "brew install"),
    ("brew",
     re.compile(r"\bdnf\s+install\b"), "brew install"),
    ("brew",
     re.compile(r"\bsudo\s+dnf\s+install\b"), "brew install"),
    ("brew",
     re.compile(r"\bapk\s+add\b"), "brew install"),
    ("brew",
     re.compile(r"\bsudo\s+apk\s+add\b"), "brew install"),
]


def postprocess_command(cmd: str, pkg_mgr: str) -> str:
    """Rewrite a command from the wrong distro's syntax to the host's.

    Also strips Debian/Ubuntu flags that have no equivalent on the host distro
    (e.g. `-y`, `-qq`, `--no-install-recommends`), but ONLY when an apt/legacy
    command was actually rewritten above. Leaving a correctly-emitted native
    command untouched avoids corrupting things like `grep -qq` or `dnf -y`.
    """
    rewritten = False
    for host_pkg, pattern, replacement in PKG_MGR_REWRITE:
        if host_pkg == pkg_mgr:
            new = pattern.sub(replacement, cmd)
            if new != cmd:
                rewritten = True
            cmd = new
    if rewritten:
        cmd = _DEB_FLAGS.sub(" ", cmd)
    # Collapse any double-spaces left by the substitution.
    cmd = re.sub(r"  +", " ", cmd).strip()
    return cmd


# GBNF grammars that constrain the model to emit only commands using the host's
# package manager. This is a hard constraint — the model cannot output the
# wrong installer even if its training bias pulls it that direction.
# The grammar is deliberately permissive for everything except the install
# verb, so non-install queries (find, ls, df, etc.) are unaffected.
PKG_MGR_GRAMMARS: dict[str, str] = {
    # The grammar only constrains the install verb. Everything else is
    # unconstrained so pipes, redirects, wildcards, braces, etc. all work.
    "pacman": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo_install | install | other_cmd
sudo_install ::= "sudo " pacman_install
pacman_install ::= "pacman -S " pkg_name (" " pkg_name)*
install      ::= "pacman -S " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "apt": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo_install | install | other_cmd
sudo_install ::= "sudo " apt_install
apt_install  ::= ("apt " | "apt-get ") "install " pkg_name (" " pkg_name)*
install      ::= ("apt " | "apt-get ") "install " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "dnf": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo_install | install | other_cmd
sudo_install ::= "sudo " dnf_install
dnf_install  ::= "dnf install " pkg_name (" " pkg_name)*
install      ::= "dnf install " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "apk": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install | other_cmd
install      ::= "apk add " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "brew": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install | other_cmd
install      ::= "brew install " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "zypper": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo_install | install | other_cmd
sudo_install ::= "sudo " zypper_install
zypper_install ::= "zypper install " pkg_name (" " pkg_name)*
install      ::= "zypper install " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
    "xbps": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install | other_cmd
install      ::= "xbps-install " pkg_name (" " pkg_name)*
pkg_name     ::= [a-zA-Z0-9_-]+
other_cmd    ::= char+
char         ::= [^\n]
''',
}


def grammar_for_pkg(pkg_mgr: str) -> str | None:
    """Return a GBNF grammar string for the given package manager, or None."""
    return PKG_MGR_GRAMMARS.get(pkg_mgr)


def build(prompt: str, enabled: bool = True, cwd: Path | None = None) -> tuple[str, str]:
    """Return (system_prompt, user_message) with context folded in."""
    if not enabled:
        return cfg_mod.SYSTEM_PROMPT, prompt
    system = cfg_mod.SYSTEM_PROMPT + "\n\n" + stable_block()
    user = volatile_block(cwd) + "\n\n<request>\n" + prompt + "\n</request>"
    return system, user
