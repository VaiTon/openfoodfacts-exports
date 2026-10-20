"""Check freshness and size of published exports produced by this service."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DAILY_MAX_AGE = timedelta(hours=36)
WEEKLY_MAX_AGE = timedelta(days=8)
MAX_SIZE = 50 * 1024**3

# (flavor, repo, file, minimum size); thresholds are broad checks for empty or
# truncated outputs, not expected-size estimates.
EXPECTED_HF_EXPORTS = (
    ("OFF", "openfoodfacts/product-database", "food.parquet", 1 * 1024**3),
    ("OBF", "openfoodfacts/product-database", "beauty.parquet", 10 * 1024**2),
    ("OPFF", "openfoodfacts/product-database", "petfood.parquet", 1 * 1024**2),
    ("Open Prices", "openfoodfacts/open-prices", "prices.parquet", 1 * 1024**2),
)

# (name, S3 object key, minimum size, maximum age). The historical events dump is
# generated weekly; the other listed objects are refreshed by the daily OFF jobs.
EXPECTED_S3_EXPORTS = (
    (
        "OFF CSV EN gzip",
        "en.openfoodfacts.org.products.csv.gz",
        100 * 1024**2,
        DAILY_MAX_AGE,
    ),
    (
        "OFF CSV EN zstd",
        "en.openfoodfacts.org.products.csv.zst",
        100 * 1024**2,
        DAILY_MAX_AGE,
    ),
    (
        "OFF CSV FR gzip",
        "fr.openfoodfacts.org.products.csv.gz",
        100 * 1024**2,
        DAILY_MAX_AGE,
    ),
    (
        "OFF CSV FR zstd",
        "fr.openfoodfacts.org.products.csv.zst",
        100 * 1024**2,
        DAILY_MAX_AGE,
    ),
    (
        "OFF mobile app dump",
        "openfoodfacts-mobile-dump-products.tsv.gz",
        10 * 1024**2,
        DAILY_MAX_AGE,
    ),
    (
        "Historical events dump",
        "openfoodfacts_historical_events.jsonl.gz",
        1 * 1024**2,
        WEEKLY_MAX_AGE,
    ),
)

S3_DATASET_BASE_URL = "https://openfoodfacts-ds.s3.eu-west-3.amazonaws.com"

# Detect whether we are running in GitHub Actions, so we can emit annotations.
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def print_error(message: str) -> None:
    """Print a GitHub annotation in Actions and a simple marker elsewhere."""
    if IS_GITHUB_ACTIONS:
        # Escape workflow-command control characters in remotely sourced messages.
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{message}")
    else:
        print(f"[!] {message}")


@dataclass
class ErrorReporter:
    count: int = 0

    def report(self, message: str) -> None:
        print_error(message)
        self.count += 1


def _file_tree_url(repository: str) -> str:
    repo = quote(repository, safe="/")
    return (
        f"https://huggingface.co/api/datasets/{repo}/tree/main"
        "?recursive=true&expand=true"
    )


def _get_file_tree(repository: str) -> list[dict]:
    url = _file_tree_url(repository)
    request = Request(url, headers={"User-Agent": "openfoodfacts-exports-ci/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("Hugging Face returned an unexpected repository tree")
    return payload


def _last_commit_date(item: dict) -> datetime | None:
    commit = item.get("lastCommit") or item.get("last_commit")
    if not isinstance(commit, dict):
        return None
    raw_date = commit.get("date")
    if not isinstance(raw_date, str):
        return None
    return datetime.fromisoformat(raw_date.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    return f"{size / 1024**2:.1f} MiB"


def _check_size_and_age(
    name: str,
    url: str,
    size: int,
    updated: datetime | None,
    minimum_size: int,
    maximum_age: timedelta,
    now: datetime,
    reporter: ErrorReporter,
) -> None:
    context = f"{name} ({url})"
    failed = False
    if size < minimum_size:
        reporter.report(
            f"{context}: suspiciously small ({_format_size(size)}; "
            f"minimum {_format_size(minimum_size)})"
        )
        failed = True
    elif size > MAX_SIZE:
        reporter.report(
            f"{context}: suspiciously large ({_format_size(size)}; "
            f"maximum {_format_size(MAX_SIZE)})"
        )
        failed = True

    if updated is None:
        reporter.report(f"{context}: update time is unavailable")
        failed = True
    else:
        age = now - updated
        if age > maximum_age:
            reporter.report(
                f"{context}: last updated {age} ago "
                f"(maximum allowed: {maximum_age})"
            )
            failed = True
        elif age < timedelta(minutes=-5):
            reporter.report(f"{context}: has a future update timestamp")
            failed = True
    if not failed and updated is not None:
        age = now - updated
        print(f"{name}: {_format_size(size)}, updated {age}")


def _check_hugging_face_exports(
    now: datetime, reporter: ErrorReporter
) -> None:
    files_by_repo: dict[str, dict[str, dict]] = {}
    for _, repository, _, _ in EXPECTED_HF_EXPORTS:
        if repository in files_by_repo:
            continue
        try:
            files_by_repo[repository] = {
                item.get("path"): item for item in _get_file_tree(repository)
            }
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            reporter.report(
                f"Could not read Hugging Face file list for {repository} "
                f"({_file_tree_url(repository)}): {exc}"
            )

    for flavor, repository, path, minimum_size in EXPECTED_HF_EXPORTS:
        files = files_by_repo.get(repository)
        if files is None:
            continue
        item = files.get(path)
        name = flavor
        url = (
            f"https://huggingface.co/datasets/{quote(repository, safe='/')}"
            f"/resolve/main/{quote(path, safe='/')}"
        )
        if item is None:
            reporter.report(f"{name} ({url}): expected file is missing")
            continue

        # Use LFS size as a fallback for API responses that omit the logical size.
        lfs = item.get("lfs") or {}
        size = item.get("size") or lfs.get("size")
        if not isinstance(size, int):
            reporter.report(f"{name} ({url}): Hugging Face did not report a valid size")
            continue
        try:
            updated = _last_commit_date(item)
        except ValueError:
            updated = None
        _check_size_and_age(
            name,
            url,
            size,
            updated,
            minimum_size,
            DAILY_MAX_AGE,
            now,
            reporter,
        )


def _check_s3_exports(now: datetime, reporter: ErrorReporter) -> None:
    for name, object_key, minimum_size, maximum_age in EXPECTED_S3_EXPORTS:
        url = f"{S3_DATASET_BASE_URL}/{quote(object_key)}"
        request = Request(
            url,
            method="HEAD",
            headers={"User-Agent": "openfoodfacts-exports-ci/1.0"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                size_header = response.headers.get("Content-Length")
                modified_header = response.headers.get("Last-Modified")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            reporter.report(
                f"{name} ({url}): could not read S3 object metadata: {exc}"
            )
            continue

        try:
            size = int(size_header) if size_header is not None else None
        except ValueError:
            size = None
        if size is None:
            reporter.report(f"{name} ({url}): S3 did not report a valid content length")
            continue

        try:
            updated = (
                parsedate_to_datetime(modified_header).astimezone(timezone.utc)
                if modified_header
                else None
            )
        except (TypeError, ValueError):
            updated = None
        _check_size_and_age(
            name,
            url,
            size,
            updated,
            minimum_size,
            maximum_age,
            now,
            reporter,
        )


def main() -> int:
    now = datetime.now(timezone.utc)
    reporter = ErrorReporter()
    _check_hugging_face_exports(now, reporter)
    _check_s3_exports(now, reporter)

    if reporter.count:
        noun = "error" if reporter.count == 1 else "errors"
        print(f"{reporter.count} {noun}")
        return 1

    print("Published exports passed freshness and size checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
