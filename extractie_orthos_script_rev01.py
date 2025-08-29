#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
from shapely.geometry import mapping, box as sbox
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS

import rasterio
from rasterio.mask import mask

from owslib.wcs import WebCoverageService
from sqlalchemy import create_engine, text

# --- nieuw: voor multipart parsing ---
from email import message_from_bytes
from email.policy import default as email_default_policy

# ----------------------- CONFIG -----------------------
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://postgres:postgis@localhost:5432/ventilus")
TABLE   = os.getenv("PG_TABLE", "ortho_analyse.masten_ondergronds_merge_dissolve")
GEOM_COL= os.getenv("PG_GEOM", "geom")
NAME_COL= os.getenv("PG_NAME", "naam")

WCS_SERVICES = [
    {"alias": "WCS_A", "url": "https://geo.api.vlaanderen.be/okz/wcs"},
    {"alias": "WCS_B", "url": "https://geo.api.vlaanderen.be/omw/wcs"},
]

# Vlaanderen (Lambert 72)
REQUEST_CRS = os.getenv("REQUEST_CRS", "EPSG:31370")
BBOX_BUFFER = float(os.getenv("BBOX_BUFFER", "0.0"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
OVERWRITE  = os.getenv("OVERWRITE", "false").lower() == "true"
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# ------------------------------------------------------
OUTPUT_DIR = Path(r"C:/Users/d08909/Documents/01_WERKMAP/HISTORISCH_ORTHO/3_RESULTAAT")

def sanitize_filename(s: str) -> str:
    s = (s or "").strip().replace(os.sep, "_")
    s = re.sub(r"[^\w\-. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:200] if len(s) > 200 else s


def _two_digit_to_year(d2: str) -> int:
    n = int(d2)
    # Heuristiek orthofoto’s: 00–70 ⇒ 2000–2070, 71–99 ⇒ 1971–1999
    return (1900 + n) if n >= 71 else (2000 + n)

def extract_year_label(s: str) -> Optional[str]:
    if not s:
        return None
    # 4-cijferig bereik, bv. 1979-1990 of 1979–1990
    m = re.search(r"(?<!\d)((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})(?!\d)", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Enkel 4-cijferig jaar, bv. 1971
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", s)
    if m:
        return m.group(1)
    # 2-cijferig bereik met - of _, bv. 79-90, 08_11, 00-03
    m = re.search(r"(?<!\d)(\d{2})\s*[-_]\s*(\d{2})(?!\d)", s)
    if m:
        y1 = _two_digit_to_year(m.group(1))
        y2 = _two_digit_to_year(m.group(2))
        return f"{y1}-{y2}"
    # Enkel 2-cijferig jaar ergens in de string, bv. 24 → 2024
    m = re.search(r"(?<!\d)(\d{2})(?!\d)", s)
    if m:
        return str(_two_digit_to_year(m.group(1)))
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_targets() -> gpd.GeoDataFrame:
    engine = create_engine(PG_URL)
    cols = f'"{NAME_COL}", "{GEOM_COL}"'
    sql = f"SELECT {cols} FROM {TABLE};"
    logging.info(f"Inladen PostGIS: {sql}")
    with engine.connect() as conn:
        gdf = gpd.read_postgis(text(sql), conn, geom_col=GEOM_COL)
    if gdf.empty:
        logging.error("Geen rijen gevonden in de PostGIS-tabel.")
        sys.exit(1)
    if gdf.crs is None:
        logging.warning("Geen CRS op geometrie; veronderstel EPSG:31370.")
        gdf.set_crs(epsg=31370, inplace=True)
    return gdf


def get_wcs_client(url: str):
    try:
        w = WebCoverageService(url, version='2.0.1')
        _ = list(w.contents.keys())
        return w, '2.0.1'
    except Exception as e:
        logging.info(f"WCS 2.0.1 faalde ({e}); probeer 1.0.0")
    w = WebCoverageService(url, version='1.0.0')
    _ = list(w.contents.keys())
    return w, '1.0.0'


def list_coverages(wcs, version: str, filter_regex: Optional[str]) -> List[Dict]:
    patt = re.compile(filter_regex, re.IGNORECASE) if filter_regex else None
    items = []
    for cov_id, cov in wcs.contents.items():
        title = getattr(cov, "title", "") or str(cov_id)
        ident = str(getattr(cov, "id", cov_id))
        if patt and not (patt.search(ident) or patt.search(title)):
            continue
        items.append({"identifier": ident, "title": title})
    logging.info(f"{len(items)} coverages geselecteerd na filter.")
    return items


def pick_tiff_format(cov) -> str:
    fmts = []
    for attr in ("formats", "supportedFormats"):
        v = getattr(cov, attr, None)
        if v:
            fmts = list(v)
            break
    for cand in fmts:
        s = str(cand).lower()
        if "geotiff" in s or "image/tiff" in s or "tiff" in s:
            return cand
    return "image/tiff"


def geom_or_bboxpoly(geom, eps=0.05):
    try:
        gt = geom.geom_type
    except Exception:
        gt = ""
    if gt in ("Polygon", "MultiPolygon"):
        return geom
    minx, miny, maxx, maxy = geom.bounds
    if maxx == minx:
        maxx += eps; minx -= eps
    if maxy == miny:
        maxy += eps; miny -= eps
    return sbox(minx, miny, maxx, maxy)


def _read_response_bytes(resp) -> bytes:
    content = None
    if hasattr(resp, "read") and callable(resp.read):
        content = resp.read()
    elif hasattr(resp, "content"):
        content = resp.content
    elif isinstance(resp, (bytes, bytearray)):
        content = resp
    if not content:
        raise RuntimeError("Lege WCS-respons (geen content).")
    return content


def _content_type(resp) -> Optional[str]:
    if hasattr(resp, "headers"):
        return resp.headers.get("Content-Type")
    if hasattr(resp, "info"):
        try:
            return resp.info().get("Content-Type")
        except Exception:
            return None
    return None


def _extract_tiff_from_multipart(raw: bytes, ct_header: Optional[str]) -> Optional[bytes]:
    """Parseer multipart/related en haal de image/tiff part eruit."""
    if not ct_header or "multipart" not in ct_header.lower():
        return None
    # Bouw een synthetische MIME-header zodat email.parser het kan lezen
    synthetic = f"Content-Type: {ct_header}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw
    msg = message_from_bytes(synthetic, policy=email_default_policy)
    for part in msg.walk():
        ctype = (part.get_content_type() or "").lower()
        if "tiff" in ctype:  # image/tiff of image/geotiff
            return part.get_payload(decode=True)
    return None


def _wcs20_get(wcs, coverage_id: str, fmt: str, req_crs: str,
               bbox: Tuple[float, float, float, float]):
    """WCS 2.0.1: probeer combinaties van aslabels en (subsetting/output) CRS."""
    minx, miny, maxx, maxy = bbox
    axes_variants = [('E', 'N'), ('x', 'y')]
    flag_variants = [(True, True), (True, False), (False, False)]
    last_err = None
    for axX, axY in axes_variants:
        for use_subcrs, use_outcrs in flag_variants:
            kwargs = {
                "identifier": [coverage_id],
                "format": fmt,
                "subsets": [(axX, float(minx), float(maxx)),
                            (axY, float(miny), float(maxy))],
            }
            if use_subcrs:
                kwargs["subsettingcrs"] = req_crs
            if use_outcrs:
                kwargs["outputcrs"] = req_crs
            try:
                logging.debug(f"GetCoverage 2.0.1 try: axes={axX}/{axY}, subcrs={use_subcrs}, outcrs={use_outcrs}")
                return wcs.getCoverage(**kwargs)
            except Exception as e:
                last_err = e
    raise last_err if last_err else RuntimeError("WCS 2.0.1: onbekende fout.")


def wcs_get_and_clip(
    wcs, version: str, coverage_id: str, req_crs: str,
    bbox: Tuple[float, float, float, float],
    clip_geom, out_path: Path, fmt: str
) -> bool:
    tmp_path = out_path.with_suffix(".tmp.tif")
    try:
        if version.startswith("2"):
            resp = _wcs20_get(wcs, coverage_id, fmt, req_crs, bbox)
        else:
            minx, miny, maxx, maxy = bbox
            resp = wcs.getCoverage(
                identifier=coverage_id,
                bbox=(minx, miny, maxx, maxy),
                crs=req_crs,
                format="GeoTIFF"
            )

        ct = _content_type(resp)
        raw = _read_response_bytes(resp)

        # 1) multipart? -> tiff eruit halen
        tif_bytes = _extract_tiff_from_multipart(raw, ct)
        if tif_bytes is None:
            # 2) direct tiff?
            if ct and "tiff" in ct.lower():
                tif_bytes = raw
            else:
                # 3) enkel XML terug? check ExceptionReport of GML zonder tiff
                if b"ExceptionReport" in raw or b"<ows:Exception" in raw:
                    snippet = raw[:2048].decode("utf-8", "ignore")
                    raise RuntimeError(f"WCS ExceptionReport:\n{snippet}")
                # GML coverage zonder multipart (zeldzaam) -> probeer toch TIFF-part te vinden mislukt
                snippet = raw[:2048].decode("utf-8", "ignore")
                raise RuntimeError(f"Onverwacht XML/GML zonder TIFF-part. ContentType={ct}\n{snippet}")
        
        with open(tmp_path, "wb") as f:
            f.write(tif_bytes)

        with rasterio.open(tmp_path) as src:
            raster_crs = src.crs.to_string() if src.crs else req_crs
            poly = geom_or_bboxpoly(clip_geom)
            if raster_crs and raster_crs != req_crs:
                transformer = Transformer.from_crs(CRS.from_user_input(req_crs), CRS.from_user_input(raster_crs), always_xy=True)
                poly = shp_transform(transformer.transform, poly)

            out_arr, out_transform = mask(src, [mapping(poly)], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_arr.shape[1],
                "width": out_arr.shape[2],
                "transform": out_transform
            })

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_arr)

        return True

    except Exception as e:
        logging.warning(f"Faalde op {coverage_id}: {e}")
        return False
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = read_targets()
    gdf_req = gdf.to_crs(REQUEST_CRS)

    all_services = []
    for svc in WCS_SERVICES:
        alias = svc["alias"]; url = svc["url"]
        filter_regex = svc.get("filter_regex")
        logging.info(f"Verbind met {alias}: {url}")
        wcs, version = get_wcs_client(url)
        covs = list_coverages(wcs, version, filter_regex)
        all_services.append({"alias": alias, "url": url, "wcs": wcs, "version": version, "coverages": covs})
        logging.info(f"{alias} ({version}): {len(covs)} coverages")

    for idx, row in gdf_req.iterrows():
        name_raw = str(row[NAME_COL])
        name = sanitize_filename(name_raw) or f"feat_{idx}"
        geom = row[GEOM_COL]
        if geom is None or geom.is_empty:
            logging.info(f"Rij {idx}: lege geometrie; sla over")
            continue

        minx, miny, maxx, maxy = geom.bounds
        if BBOX_BUFFER > 0:
            minx -= BBOX_BUFFER; miny -= BBOX_BUFFER
            maxx += BBOX_BUFFER; maxy += BBOX_BUFFER
        bbox = (minx, miny, maxx, maxy)

        out_dir = OUTPUT_DIR / name
        ensure_dir(out_dir)

        logging.info(f"Feature '{name_raw}' -> {out_dir.name} | bbox {REQUEST_CRS}: {tuple(round(v,3) for v in bbox)}")

        for svc in all_services:
            alias   = svc["alias"]
            wcs     = svc["wcs"]
            version = svc["version"]
            for cov in svc["coverages"]:
                cov_id    = cov["identifier"]
                cov_title = cov["title"]
                cov_obj   = wcs.contents[cov_id]
                fmt       = pick_tiff_format(cov_obj)
                label = extract_year_label(cov_id) or extract_year_label(cov_title) or "unknown"
                out_path = out_dir / f"{label}.tif"
                
                if out_path.exists() and not OVERWRITE:
                    stem = out_path.stem; k = 2
                    candidate = out_dir / f"{stem}_{alias}.tif"
                    if candidate.exists():
                        while True:
                            candidate = out_dir / f"{stem}_{alias}_{k}.tif"
                            if not candidate.exists():
                                break
                            k += 1
                    out_path = candidate

                ok = wcs_get_and_clip(
                    wcs=wcs,
                    version=version,
                    coverage_id=cov_id,
                    req_crs=REQUEST_CRS,
                    bbox=bbox,
                    clip_geom=geom,   # geom al in REQUEST_CRS
                    out_path=out_path,
                    fmt=fmt
                )
                if ok:
                    logging.info(f"✔ {alias}:{cov_id} -> {out_path.relative_to(OUTPUT_DIR)}")
                else:
                    logging.info(f"✖ {alias}:{cov_id} (overgeslagen)")

                time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    logging.info("Klaar.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Afgebroken door gebruiker.")
