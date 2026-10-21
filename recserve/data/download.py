"""Download MovieLens 25M, verify its MD5 and convert each CSV to Parquet.

CSVs are streamed straight out of the zip into Parquet, so the ~1.2 GB of
uncompressed CSV never touches disk. Usage::

    python -m recserve.data.download --config configs/default.yaml
"""

from __future__ import annotations

import hashlib
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import truststore

from recserve.config import config_from_cli
from recserve.utils import log, setup_logging, timed

# Explicit column types keep memory low (int32 ids, float32 ratings) and make
# the schema independent of pyarrow's type inference.
SCHEMAS: dict[str, dict[str, pa.DataType]] = {
    "ratings": {
        "userId": pa.int32(),
        "movieId": pa.int32(),
        "rating": pa.float32(),
        "timestamp": pa.int64(),
    },
    "movies": {"movieId": pa.int32(), "title": pa.string(), "genres": pa.string()},
    "tags": {
        "userId": pa.int32(),
        "movieId": pa.int32(),
        "tag": pa.string(),
        "timestamp": pa.int64(),
    },
    "links": {"movieId": pa.int32(), "imdbId": pa.string(), "tmdbId": pa.float64()},
    "genome-scores": {"movieId": pa.int32(), "tagId": pa.int16(), "relevance": pa.float32()},
    "genome-tags": {"tagId": pa.int16(), "tag": pa.string()},
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# Verify TLS against the OS trust store instead of the bundled CA file, which
# fails on some machines (e.g. when the server omits an intermediate cert).
truststore.inject_into_ssl()


def fetch(url: str, dest: Path) -> None:
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)


def expected_md5(md5_url: str) -> str:
    with urllib.request.urlopen(md5_url) as resp:
        text = resp.read().decode()
    match = re.search(r"\b[0-9a-f]{32}\b", text)
    if not match:
        raise ValueError(f"no md5 found at {md5_url}: {text!r}")
    return match.group(0)


def convert(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".csv")}
        for stem, schema in SCHEMAS.items():
            dest = out_dir / f"{stem.replace('-', '_')}.parquet"
            with timed(f"convert {stem}"), zf.open(names[stem]) as f:
                table = pacsv.read_csv(f, convert_options=pacsv.ConvertOptions(column_types=schema))
                pq.write_table(table, dest, compression="zstd")
            log.info("%s: %d rows -> %s", stem, table.num_rows, dest)


def main() -> None:
    setup_logging()
    cfg = config_from_cli(__doc__.splitlines()[0])
    dcfg = cfg["data"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{dcfg['dataset']}.zip"

    if not zip_path.exists():
        fetch(dcfg["url"], zip_path)
    want = expected_md5(dcfg["md5_url"])
    got = md5sum(zip_path)
    if got != want:
        raise RuntimeError(f"MD5 mismatch for {zip_path}: expected {want}, got {got}")
    log.info("md5 ok (%s)", got)

    convert(zip_path, Path(cfg["paths"]["processed_dir"]))
    if not dcfg.get("keep_raw", False):
        zip_path.unlink()
        log.info("deleted %s (set data.keep_raw=true to keep it)", zip_path)


if __name__ == "__main__":
    main()
