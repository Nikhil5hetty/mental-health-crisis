#!/usr/bin/env python3
"""Fetch published bytes without overwriting raw files. Run with python -I.

MANIFEST.json is both provenance and the revision/checksum lock. A first run
resolves upstream revisions; subsequent runs use those exact revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
import zipfile

if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]])

import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/MANIFEST.json"
STAGING = ROOT / "data/.staging"
LFS_PREFIX = b"version https://git-lfs"
MHB_PAGE = "https://openai.com/index/introducing-mentalhealthbench/"
MHB_PAPER = "https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf"
# Official paper, section 8 (Data Access), linked from MHB_PAGE.
MHB_ARCHIVE = "https://cdn.openai.com/ctf-cdn/OAI_MentalHealthBench.zip"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Fetcher:
    def __init__(self):
        self.manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
            "schema_version": 1, "sources": {}, "datasets": {}, "files": []}
        self.entries = {f["path"]: f for f in self.manifest["files"]}
        self.downloads = 0
        self.created = 0
        self.skipped = 0

    def save(self):
        self.manifest["files"] = sorted(self.entries.values(), key=lambda f: f["path"])
        content = json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n"
        if MANIFEST.exists() and MANIFEST.read_text() == content:
            return
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        temp = MANIFEST.with_suffix(".json.tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(MANIFEST)

    def check_path(self, path):
        if not path.resolve().is_relative_to((ROOT / "data").resolve()):
            raise ValueError(f"Path outside data/: {path}")
        for p in (path, *path.parents):
            if p == ROOT:
                break
            if p.is_symlink():
                raise ValueError(f"Refusing symlink: {p}")

    def existing(self, path):
        self.check_path(path)
        entry = self.entries.get(path.relative_to(ROOT).as_posix())
        if path.exists() and entry:
            if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch; preserved without overwrite: {path}")
            self.skipped += 1
            return True
        return False

    def install(self, src, dest, dataset, source, url, member=None):
        self.check_path(dest)
        if src.is_symlink():
            raise ValueError(f"Refusing source symlink: {src}")
        with src.open("rb") as stream:
            if stream.read(64).startswith(LFS_PREFIX):
                raise ValueError(f"Unresolved Git LFS pointer: {src}")
        digest = sha256(src)
        key = dest.relative_to(ROOT).as_posix()
        old = self.entries.get(key)
        if old and (digest != old["sha256"] or src.stat().st_size != old["size"]):
            raise ValueError(f"Source differs from manifest; refusing: {dest}")
        if dest.exists():
            if sha256(dest) != digest:
                raise ValueError(f"Existing raw file differs; preserved: {dest}")
            self.skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Stage bytes and publish with an exclusive hard link; never replace raw.
            STAGING.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=STAGING) as stage:
                temp = Path(stage) / "file"
                shutil.copyfile(src, temp)
                os.link(temp, dest)
            self.created += 1
        if not old:
            self.entries[key] = {
                "path": key, "dataset": dataset, "source": source,
                "source_url": url,
                "revision": self.manifest["sources"][source]["revision"],
                "download_date_utc": self.manifest["sources"][source]["download_date_utc"],
                "size": src.stat().st_size, "sha256": digest,
                "licence": self.manifest["sources"][source]["licence"],
            }
            if member is not None:
                self.entries[key]["archive_member"] = member

    def git_source(self, name, url, licence):
        dest = ROOT / "data/external_repos" / name
        self.check_path(dest)
        sources = self.manifest["sources"]
        locked = sources.get(name)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            git("clone", url, str(dest))
            self.downloads += 1
            if locked:
                git("-C", str(dest), "checkout", "--detach", locked["revision"])
        if git("-C", str(dest), "remote", "get-url", "origin").removesuffix(".git") != url.removesuffix(".git"):
            raise ValueError(f"Unexpected origin: {dest}")
        revision = git("-C", str(dest), "rev-parse", "HEAD")
        if locked and revision != locked["revision"]:
            raise ValueError(f"Revision differs from manifest: {dest}")
        if git("-C", str(dest), "status", "--porcelain"):
            raise ValueError(f"Source checkout is modified; refusing: {dest}")
        paths = [dest / p for p in git("-C", str(dest), "ls-files", "-z").split("\0") if p]
        pointers = []
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Source symlink: {path}")
            with path.open("rb") as stream:
                if stream.read(64).startswith(LFS_PREFIX):
                    pointers.append(path.relative_to(dest).as_posix())
        if pointers:
            print(f"Git LFS pointers in {name}: {pointers}; running git lfs pull for this repo only", flush=True)
            git("-C", str(dest), "lfs", "pull")
            self.downloads += 1
            for path in paths:
                with path.open("rb") as stream:
                    if stream.read(64).startswith(LFS_PREFIX):
                        raise ValueError(f"Unresolved LFS pointer: {path}")
        if not locked:
            sources[name] = {"type": "git", "url": url, "revision": revision,
                             "download_date_utc": utc_now(), "licence": licence,
                             "lfs_pointers_resolved": pointers}
            self.save()
        print(f"Source {name}: {revision}; LFS pointers this run: {len(pointers)}", flush=True)
        return dest, paths

    def copy_git(self, source, paths, prefix, target, dataset):
        info = self.manifest["sources"][source]
        repo = ROOT / "data/external_repos" / source
        for path in paths:
            relative = path.relative_to(prefix)
            dest = target / relative
            if self.existing(dest):
                continue
            url = f'{info["url"]}/blob/{info["revision"]}/{quote(path.relative_to(repo).as_posix())}'
            self.install(path, dest, dataset, source, url)

    def extract_zip(self, archive, target, dataset, source):
        entry = self.entries[archive.relative_to(ROOT).as_posix()]
        with zipfile.ZipFile(archive) as zf:
            seen = set()
            # Validate all names and file types before extracting any member.
            for item in zf.infolist():
                p = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                if (p.is_absolute() or ".." in p.parts or "\\" in item.filename
                        or ":" in item.filename or not p.parts
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError(f"Unsafe archive member: {item.filename}")
                if p.as_posix() in seen:
                    raise ValueError(f"Duplicate archive member: {item.filename}")
                seen.add(p.as_posix())
                self.check_path(target / p)
            STAGING.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=STAGING) as stage:
                for item in zf.infolist():
                    dest = target / item.filename
                    if item.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        continue
                    if self.existing(dest):
                        continue
                    temp = Path(stage) / "member"
                    with zf.open(item) as src, temp.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    self.install(temp, dest, dataset, source, entry["source_url"], item.filename)


def git(*args):
    # Never run downloaded hooks or checkout filters automatically.
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false",
               "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=",
               "-c", "filter.lfs.required=false", *args]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout.strip("\n")


def fetch_between_help_and_harm(f):
    name = "between_help_and_harm"
    repo, paths = f.git_source(name, "https://github.com/ellisalicante/LLMs-Mental-Health-Crisis",
                              "Not specified in source repository")
    paths = [p for p in paths if p.is_relative_to(repo / "data")]
    if not paths:
        raise ValueError("Between Help and Harm data/ is empty")
    f.copy_git(name, paths, repo / "data", ROOT / "data/raw" / name, name)


def fetch_counselbench(f):
    f.git_source("counselbench_code", "https://github.com/llm-eval-mental-health/CounselBench",
                 "Not specified in code repository; HF datasets: CC-BY-NC-ND-4.0")
    token = os.environ.get("HF_TOKEN") or False
    for name in ("CounselBench-Eval", "CounselBench-Adv", "CounselBench-Adv-human-annotation"):
        repo_id = "izi-ano/" + name
        target = ROOT / "data/raw/counselbench" / name
        sources = f.manifest["sources"]
        if repo_id not in sources:
            info = HfApi(token=token).dataset_info(repo_id, files_metadata=True)
            if info.gated and not token:
                raise GatedRepoError(f"{repo_id} is gated: set HF_TOKEN after obtaining access")
            sources[repo_id] = {"type": "huggingface", "url": f"https://huggingface.co/datasets/{repo_id}",
                "revision": info.sha, "download_date_utc": utc_now(), "licence": "CC-BY-NC-ND-4.0",
                "files": [s.rfilename for s in info.siblings]}
            f.save()
        info = sources[repo_id]
        missing = [name for name in info["files"] if not f.existing(target / name)]
        if not missing:
            print(f"SKIP {repo_id}: all files match sha256", flush=True)
            continue
        # snapshot_download writes only to staging; raw is always installed exclusively.
        STAGING.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=STAGING) as stage:
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=info["revision"],
                              token=token, local_dir=stage, allow_patterns=missing)
            f.downloads += 1
            for filename in missing:
                f.install(Path(stage) / filename, target / filename, "counselbench", repo_id,
                          f'{info["url"]}/resolve/{info["revision"]}/{quote(filename)}')
        f.save()


def fetch_mentalalign(f):
    name = "mentalbench_align"
    repo, paths = f.git_source(name, "https://github.com/abeerbadawi/MentalBench-Align", "MIT")
    selected = [p for p in paths if (
        p.is_relative_to(repo / "MentalAlign-70k") and not p.name.startswith("~$")) or (
        p.parent == repo / "Codes/Analysis" and p.name.endswith("_llm_judge_with_human.csv")) or (
        p == repo / "MentalBench-100k/MentalBench-100k.zip")]
    target = ROOT / "data/raw/mentalalign"
    f.copy_git(name, selected, repo, target, "mentalalign")
    f.extract_zip(target / "MentalBench-100k/MentalBench-100k.zip",
                  target / "MentalBench-100k_extracted", "mentalalign", name)


def fetch_mentalhealthbench(f):
    name = "mentalhealthbench"
    sources = f.manifest["sources"]
    if name not in sources:
        sources[name] = {"type": "official_archive", "url": MHB_ARCHIVE,
            "discovery_page": MHB_PAGE, "discovery_paper": MHB_PAPER,
            "discovery_section": "8 Data Access (page 23)",
            "revision": None, "revision_note": "No Git/HF revision published; archive pinned by SHA-256",
            "download_date_utc": utc_now(), "licence": "MIT"}
        f.save()
    target = ROOT / "data/raw" / name
    archive = target / "OAI_MentalHealthBench.zip"
    if not f.existing(archive):
        STAGING.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=STAGING) as stage:
            temp = Path(stage) / archive.name
            with requests.get(sources[name]["url"], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if urlparse(response.url).hostname != "cdn.openai.com":
                    raise ValueError("Official archive redirected away from cdn.openai.com; stopped")
                with temp.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if not zipfile.is_zipfile(temp):
                raise ValueError("Official download is not a ZIP archive; stopped")
            f.downloads += 1
            f.install(temp, archive, name, name, sources[name]["url"])
    f.extract_zip(archive, target, name, name)


def fetch_capphrase(f):
    name = "capphrase"
    repo, paths = f.git_source(name, "https://github.com/adamkucharski/CAPphrase", "CC-BY-4.0")
    selected = [repo / "data" / filename for filename in (
        "absolute_judgements.csv", "pairwise_comparisons.csv", "respondent_metadata.csv")]
    if not all(path in paths for path in selected):
        raise ValueError("CAPphrase is missing a required published CSV")
    f.copy_git(name, selected, repo / "data", ROOT / "data/raw" / name, name)


FETCHERS = {"between_help_and_harm": fetch_between_help_and_harm, "counselbench": fetch_counselbench,
            "mentalalign": fetch_mentalalign, "mentalhealthbench": fetch_mentalhealthbench, "capphrase": fetch_capphrase}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=FETCHERS)
    args = parser.parse_args()
    f = Fetcher()
    failures = []
    for name in [args.only] if args.only else FETCHERS:
        print(f"\nFetching {name}", flush=True)
        try:
            FETCHERS[name](f)
            f.manifest["datasets"][name] = {"status": "fetched"}
        except GatedRepoError:
            reason = "Gated Hugging Face dataset: obtain access and set HF_TOKEN; token values are never recorded"
            f.manifest["datasets"][name] = {"status": "blocked", "reason": reason}
            failures.append(name)
            print(f"FAIL {name}: {reason}", flush=True)
            break
        except (OSError, ValueError, subprocess.CalledProcessError, requests.RequestException, zipfile.BadZipFile) as exc:
            reason = str(exc)
            if os.environ.get("HF_TOKEN"):
                reason = reason.replace(os.environ["HF_TOKEN"], "[REDACTED]")
            f.manifest["datasets"][name] = {"status": "failed", "reason": reason}
            failures.append(name)
            print(f"FAIL {name}: {reason}", flush=True)
        finally:
            f.save()
    print(f"\nFetch summary: download operations={f.downloads}, new raw files={f.created}, "
          f"matching files skipped={f.skipped}, failed datasets={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
