"""Offline regression tests using only synthetic files, never the real raw data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load("fetch_data")
verify = load("verify_data")


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for attr, value in {"ROOT": self.root, "MANIFEST": self.root / "data/MANIFEST.json",
                            "STAGING": self.root / "data/.staging"}.items():
            patcher = patch.object(fetch, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.f = fetch.Fetcher()
        self.f.manifest["sources"]["fixture"] = {
            "revision": "a" * 40, "download_date_utc": "2026-11-21T00:00:00Z", "licence": "MIT"}
        self.source = self.root / "source.csv"
        self.source.write_bytes(b'col\r\n"unchanged bytes"\r\n')
        self.target = self.root / "data/raw/fixture/source.csv"

    def install(self):
        self.f.install(self.source, self.target, "fixture", "fixture", "https://example.org/source.csv")

    def test_second_install_preserves_bytes_and_manifest(self):
        self.install()
        self.f.save()
        before = fetch.MANIFEST.read_bytes()
        modified = self.target.stat().st_mtime_ns
        self.f = fetch.Fetcher()
        self.assertTrue(self.f.existing(self.target))
        self.install()
        self.f.save()
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        self.assertEqual(self.target.stat().st_mtime_ns, modified)
        self.assertEqual(fetch.MANIFEST.read_bytes(), before)
        self.assertEqual(self.f.downloads, 0)
        self.assertEqual(self.f.created, 0)

    def test_corrupt_raw_is_never_overwritten(self):
        self.install()
        self.target.write_bytes(b"fixture corruption")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.f.existing(self.target)
        with self.assertRaisesRegex(ValueError, "preserved"):
            self.install()
        self.assertEqual(self.target.read_bytes(), b"fixture corruption")

    def test_changed_download_cannot_replace_locked_source(self):
        self.install()
        original = self.target.read_bytes()
        self.source.write_bytes(b"upstream changed")
        with self.assertRaisesRegex(ValueError, "Source differs"):
            self.install()
        self.assertEqual(self.target.read_bytes(), original)

    def test_unrecorded_conflicting_raw_is_preserved(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"previous user data")
        with self.assertRaisesRegex(ValueError, "preserved"):
            self.install()
        self.assertEqual(self.target.read_bytes(), b"previous user data")

    def test_lfs_pointer_is_rejected(self):
        self.source.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            self.install()
        self.assertFalse(self.target.exists())

    def test_symlink_and_traversal_destinations_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.target.parent.mkdir(parents=True)
        (self.target.parent / "link").symlink_to(outside, target_is_directory=True)
        for destination in (self.target.parent / "link/file", self.root / "data/../outside/file"):
            with self.assertRaises(ValueError):
                self.f.install(self.source, destination, "fixture", "fixture", "https://example.org/file")
        self.assertFalse((outside / "file").exists())

    def archive(self, members):
        source = self.root / "source.zip"
        with zipfile.ZipFile(source, "w") as zf:
            for name, content in members:
                zf.writestr(name, content)
        archive = self.root / "data/raw/fixture/archive.zip"
        self.f.install(source, archive, "fixture", "fixture", "https://example.org/archive.zip")
        return archive

    def test_archive_keeps_member_bytes_and_is_idempotent(self):
        data = b"col\r\n123\r\n"
        archive = self.archive([("nested/file.csv", data)])
        target = self.root / "data/raw/fixture/extracted"
        self.f.extract_zip(archive, target, "fixture", "fixture")
        entry = self.f.entries["data/raw/fixture/extracted/nested/file.csv"]
        self.assertEqual(entry["archive_member"], "nested/file.csv")
        self.assertEqual((target / "nested/file.csv").read_bytes(), data)
        count = self.f.created
        self.f.extract_zip(archive, target, "fixture", "fixture")
        self.assertEqual(self.f.created, count)
        self.assertTrue(archive.is_file())

    def test_archive_traversal_rejected_before_extraction(self):
        archive = self.archive([("safe.csv", b"safe"), ("../escape.csv", b"unsafe")])
        target = self.root / "data/raw/fixture/extracted"
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            self.f.extract_zip(archive, target, "fixture", "fixture")
        self.assertFalse((target / "safe.csv").exists())
        self.assertFalse((target.parent / "escape.csv").exists())

    def test_archive_symlink_rejected(self):
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive = self.archive([(member, "../outside")])
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            self.f.extract_zip(archive, self.root / "data/raw/fixture/extracted", "fixture", "fixture")

    def test_missing_datasets_exit_nonzero_and_report_all_sources(self):
        result = subprocess.run([sys.executable, "-I", str(Path(verify.__file__)), "--root", str(self.root)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        for name in ("CounselBench", "Between Help and Harm", "MentalAlign", "MentalHealthBench"):
            self.assertIn(name, result.stdout)
        self.assertIn("FAIL", result.stdout)
        self.assertFalse((self.root / "data/raw").exists())


class VerificationTests(unittest.TestCase):
    def test_safety_counts_missing_null_out_of_range_and_malformed(self):
        df = verify.pd.DataFrame({" human_response_original": [
            '{"Safety": 0}', '{"Safety": 5}', '{"Safety": 6}', '{"Safety": null}',
            float("nan"), "not JSON", '{"Safety": true}']})
        counts = verify.safety_counts(df)
        self.assertEqual(counts["outside_1_to_5"], 2)
        self.assertEqual(counts["parsed"], 3)
        self.assertEqual(len(counts["missing_cells"]), 2)
        self.assertEqual(len(counts["unreadable_cells"]), 2)
        self.assertEqual(df.iloc[0, 0], '{"Safety": 0}')

    def test_report_mismatch_is_failure(self):
        report = verify.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            report.check("rows", 1999, 2000)
        self.assertEqual(report.failed, 1)


if __name__ == "__main__":
    unittest.main()
