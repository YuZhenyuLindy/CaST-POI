"""
Download NYC, TKY, and CA datasets for next-POI recommendation.

Sources used by the paper:
- NYC, TKY:  Foursquare TSMC2014 (Yang et al., 2015).
             Page: https://sites.google.com/site/yangdingqi/home/foursquare-dataset
             File: dataset_TSMC2014.zip  (contains dataset_TSMC2014_NYC.txt /
                   dataset_TSMC2014_TKY.txt)
- CA:        Gowalla check-ins (Cho et al., 2011), filtered to California.
             Page: https://snap.stanford.edu/data/loc-gowalla.html
             File: https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz

This script:
  1. Tries direct download from the canonical mirror.
  2. If that fails, prints manual-download instructions and exits non-zero.
     It does not fall back to synthetic data.
  3. Once the raw file is on disk, normalises it to a uniform CSV at
     `<data_dir>/<DATASET>/checkins.csv` with columns:
         user_id, poi_id, category, latitude, longitude, timestamp
     so that `preprocess.py` can consume it.

Usage:
  python data/download.py --dataset all
  python data/download.py --dataset ca
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------
# Direct file URLs where available. For Foursquare the official page requires a
# manual download, so community mirrors of the same TSMC2014 files are tried in
# order first.
SOURCES = {
    'nyc': {
        'manual_page': 'https://sites.google.com/site/yangdingqi/home/foursquare-dataset',
        'expected_filename': 'dataset_TSMC2014_NYC.txt',
        'mirrors': [
            # Foursquare TSMC2014 zip; we extract NYC.txt from it.
            'http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip',
        ],
    },
    'tky': {
        'manual_page': 'https://sites.google.com/site/yangdingqi/home/foursquare-dataset',
        'expected_filename': 'dataset_TSMC2014_TKY.txt',
        'mirrors': [
            'http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip',
        ],
    },
    'ca': {
        'manual_page': 'https://snap.stanford.edu/data/loc-gowalla.html',
        'expected_filename': 'Gowalla_totalCheckins.txt',
        'mirrors': [
            'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz',
        ],
    },
}

# Foursquare TSMC2014 column layout (tab-separated):
#   userId, venueId, venueCategoryId, venueCategory, latitude, longitude,
#   timezoneOffset, utcTimestamp
FOURSQUARE_COLS = [
    'user_id', 'poi_id', 'category_id', 'category',
    'latitude', 'longitude', 'tz_offset', 'utc_timestamp',
]

# Gowalla SNAP layout (tab-separated, no header):
#   userId, checkinTime, latitude, longitude, locationId
GOWALLA_COLS = ['user_id', 'utc_timestamp', 'latitude', 'longitude', 'poi_id']


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
def http_get_to_file(url: str, save_path: Path, timeout: int = 60) -> bool:
    """Stream a URL to disk with a progress bar. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = save_path.with_suffix(save_path.suffix + '.part')
            with open(tmp, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True,
                desc=save_path.name, dynamic_ncols=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            tmp.replace(save_path)
        return True
    except Exception as e:
        print(f"  ! download failed ({url}): {e}")
        if save_path.with_suffix(save_path.suffix + '.part').exists():
            save_path.with_suffix(save_path.suffix + '.part').unlink()
        return False


def gunzip_file(src: Path, dst: Path):
    with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout:
        shutil.copyfileobj(fin, fout)


# ---------------------------------------------------------------------------
# Dataset acquisition
# ---------------------------------------------------------------------------
def _acquire_foursquare(dataset_name: str, raw_dir: Path) -> Optional[Path]:
    """
    Acquire NYC or TKY Foursquare TSMC2014 .txt. Strategy:
      1. If `<raw_dir>/<DATASET>/dataset_TSMC2014_<NAME>.txt` already exists, use it.
      2. Try mirrors that serve the zip; extract the right file.
      3. Otherwise return None.
    """
    spec = SOURCES[dataset_name]
    save_dir = raw_dir / dataset_name.upper()
    target = save_dir / spec['expected_filename']
    if target.exists():
        print(f"  [{dataset_name}] using existing {target}")
        return target

    for url in spec['mirrors']:
        zip_path = save_dir / 'dataset_TSMC2014.zip'
        print(f"  [{dataset_name}] trying mirror: {url}")
        if not http_get_to_file(url, zip_path):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                # The expected member ends in dataset_TSMC2014_<NAME>.txt
                wanted = next((n for n in names
                               if n.endswith(spec['expected_filename'])), None)
                if wanted is None:
                    print(f"  ! zip did not contain {spec['expected_filename']}; "
                          f"members={names[:5]}...")
                    zip_path.unlink(missing_ok=True)
                    continue
                with zf.open(wanted) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
            zip_path.unlink(missing_ok=True)
            print(f"  [{dataset_name}] extracted -> {target}")
            return target
        except zipfile.BadZipFile as e:
            print(f"  ! corrupt zip: {e}")
            zip_path.unlink(missing_ok=True)
            continue

    return None


def _acquire_gowalla(raw_dir: Path) -> Optional[Path]:
    """Acquire Gowalla totalCheckins. Direct download from SNAP works."""
    spec = SOURCES['ca']
    save_dir = raw_dir / 'CA'
    target = save_dir / spec['expected_filename']
    if target.exists():
        print(f"  [ca] using existing {target}")
        return target

    for url in spec['mirrors']:
        gz_path = save_dir / 'loc-gowalla_totalCheckins.txt.gz'
        print(f"  [ca] trying mirror: {url}")
        if not http_get_to_file(url, gz_path):
            continue
        try:
            gunzip_file(gz_path, target)
            gz_path.unlink(missing_ok=True)
            print(f"  [ca] extracted -> {target}")
            return target
        except OSError as e:
            print(f"  ! gunzip failed: {e}")
            gz_path.unlink(missing_ok=True)
            continue

    return None


# ---------------------------------------------------------------------------
# Normalisation -> checkins.csv (consumed by preprocess.py)
# ---------------------------------------------------------------------------
def _normalise_foursquare(src_txt: Path, save_dir: Path) -> Path:
    """Parse TSMC2014 .txt and write checkins.csv."""
    print(f"  parsing {src_txt}")
    df = pd.read_csv(src_txt, sep='\t', names=FOURSQUARE_COLS,
                     encoding='latin-1', engine='python')
    # Convert UTC timestamp string to unix seconds
    ts = pd.to_datetime(df['utc_timestamp'],
                        format='%a %b %d %H:%M:%S %z %Y',
                        utc=True, errors='coerce')
    df['timestamp'] = (ts.view('int64') // 10**9).astype('int64')
    out = df[['user_id', 'poi_id', 'category',
              'latitude', 'longitude', 'timestamp']].copy()
    out = out.dropna()
    out = out[out['timestamp'] > 0]
    csv_path = save_dir / 'checkins.csv'
    out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(out):,} rows)")
    return csv_path


def _normalise_gowalla(src_txt: Path, save_dir: Path,
                       ca_lat=(32.5, 42.0), ca_lon=(-124.5, -114.0)) -> Path:
    """Parse SNAP Gowalla, filter to California bounding box, write checkins.csv."""
    print(f"  parsing {src_txt}  (filtering to California bbox)")
    df = pd.read_csv(src_txt, sep='\t', names=GOWALLA_COLS,
                     header=None, engine='c')
    in_ca = (
        df['latitude'].between(*ca_lat)
        & df['longitude'].between(*ca_lon)
    )
    df = df.loc[in_ca].copy()
    ts = pd.to_datetime(df['utc_timestamp'], utc=True, errors='coerce')
    df['timestamp'] = (ts.view('int64') // 10**9).astype('int64')
    df['category'] = 'Unknown'  # SNAP Gowalla has no categories
    out = df[['user_id', 'poi_id', 'category',
              'latitude', 'longitude', 'timestamp']].copy()
    out = out.dropna()
    out = out[out['timestamp'] > 0]
    csv_path = save_dir / 'checkins.csv'
    out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(out):,} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Main entry per dataset
# ---------------------------------------------------------------------------
def download_dataset(dataset_name: str, data_dir: str = './raw') -> Path:
    name = dataset_name.lower()
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(SOURCES)}.")

    raw_dir = Path(data_dir)
    save_dir = raw_dir / name.upper()
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / 'checkins.csv'
    if csv_path.exists():
        print(f"[{name}] {csv_path} already exists — skipping.")
        return csv_path

    print(f"\n=== Acquiring {name.upper()} ===")
    if name in ('nyc', 'tky'):
        src = _acquire_foursquare(name, raw_dir)
    else:  # ca
        src = _acquire_gowalla(raw_dir)

    if src is None:
        # Exit with manual-download instructions rather than substituting data.
        spec = SOURCES[name]
        print(
            f"\n[FATAL] could not auto-download {name.upper()}.\n"
            f"  Please download it manually:\n"
            f"    Page : {spec['manual_page']}\n"
            f"    File : {spec['expected_filename']}\n"
            f"  Then place it at:\n"
            f"    {save_dir / spec['expected_filename']}\n"
            f"  and re-run:\n"
            f"    python data/download.py --dataset {name}\n",
            file=sys.stderr,
        )
        sys.exit(2)

    if name in ('nyc', 'tky'):
        return _normalise_foursquare(src, save_dir)
    else:
        return _normalise_gowalla(src, save_dir)


def download_all(data_dir: str = './raw'):
    for name in ['nyc', 'tky', 'ca']:
        try:
            download_dataset(name, data_dir)
        except SystemExit:
            print(f"  -> {name} skipped; continuing with the others.")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download POI datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['nyc', 'tky', 'ca', 'all'])
    parser.add_argument('--data_dir', type=str, default='./raw')
    args = parser.parse_args()

    if args.dataset == 'all':
        download_all(args.data_dir)
    else:
        download_dataset(args.dataset, args.data_dir)
