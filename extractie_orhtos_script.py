import os
import re
import sys
# import math
# import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
from shapely.geometry import mapping, box
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS

import rasterio
from rasterio.mask import mask

from owslib.wcs import WebCoverageService
from sqlalchemy import create_engine, text

# ----------------------- CONFIG -----------------------
# PostGIS
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://postgres:postgis@localhost:5432/ventilus"  # <- pas aan of zet als env var
)
TABLE = os.getenv("PG_TABLE", "ortho_analyse.masten_ondergronds_merge_testing")         # schema.tabel
GEOM_COL = os.getenv("PG_GEOM", "geom")                    # geometry(MultiPolygon,31370) bv.
NAME_COL = os.getenv("PG_NAME", "naam")                    # kolom met mapnaam per rij

# WCS services (2 stuks, voeg je endpoints toe)
WCS_SERVICES = [
    {
        "alias": "WCS_A",
        "url": "https://geo.api.vlaanderen.be/okz/wcs",
        # leave version None -> script probeert 2.0.1 en dan 1.0.0
            },
    {
        "alias": "WCS_B",
        "url": "https://geo.api.vlaanderen.be/omw/wcs",
            }
]

# Welke CRS gebruiken voor de WCS-request bbox?
REQUEST_CRS = os.getenv("REQUEST_CRS", "EPSG:31370")  # pas aan naar CRS die je WCS accepteert
# Buffer (meters of CRS-eenheden) rond bbox om randartefacten te vermijden
BBOX_BUFFER = float(os.getenv("BBOX_BUFFER", "0.0"))

# Uitvoer
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP", "0.5"))  # beleefd naar server

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# ------------------------------------------------------

def sanitize_filename(s: str) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^\w\-. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:200] if len(s) > 200 else s


def extract_year(s: str) -> Optional[str]:
    m = re.search(r"(19|20)\d{2}", s)
    return m.group(0) if m else None


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
        gdf.set_crs(epsg=31370, inplace=True)  # pas aan indien nodig
    return gdf


def get_wcs_client(url: str):
    # Probeer WCS 2.0.1, val terug naar 1.0.0
    try:
        w = WebCoverageService(url, version='2.0.1')
        # Forceer fetch van contents
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
        items.append({
            "identifier": ident,
            "title": title,
        })
    logging.info(f"{len(items)} coverages geselecteerd na filter.")
    return items


def reproject_bbox(bbox: Tuple[float, float, float, float], src: str, dst: str) -> Tuple[float, float, float, float]:
    if src == dst:
        return bbox
    transformer = Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)
    minx, miny, maxx, maxy = bbox
    x1, y1 = transformer.transform(minx, miny)
    x2, y2 = transformer.transform(maxx, maxy)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def feature_bbox_in_crs(geom, dst_crs: str) -> Tuple[float, float, float, float]:
    src_crs = geom.crs.to_string() if hasattr(geom, "crs") and geom.crs else None
    # geom hier is shapely; we hebben de CRS via de GeoSeries/GDF nodig
    # workaround: we gaan ervan uit dat caller de geometrie door gdf.to_crs() stuurt
    b = geom.bounds  # minx, miny, maxx, maxy
    return b


def wcs_get_and_clip(
    wcs, version: str, coverage_id: str, req_crs: str,
    bbox: Tuple[float, float, float, float],
    clip_geom, out_path: Path
) -> bool:
    """
    Haal GeoTIFF via WCS (bbox-subset), clip op polygon, schrijf naar out_path.
    """
    tmp_path = out_path.with_suffix(".tmp.tif")
    try:
        if version.startswith("2"):
            minx, miny, maxx, maxy = bbox
            subsets = [('E', f"{minx},{maxx}"), ('N', f"{miny},{maxy}")]
            # Gebruik identifier=[...] (lijst) en subsettingcrs
            try:
                resp = wcs.getCoverage(
                    identifier=[coverage_id],
                    format=fmt,                 # zie pick_tiff_format
                    subsets=subsets,
                    subsettingcrs=req_crs,      # belangrijk bij 2.0.1
                    outputcrs=req_crs           # optioneel; laat gerust weg als native goed is
                )
            except Exception:
                # fallback voor servers die x/y ipv E/N verwachten
                subsets = [('x', f"{minx},{maxx}"), ('y', f"{miny},{maxy}")]
                resp = wcs.getCoverage(
                    identifier=[coverage_id],
                    format=fmt,
                    subsets=subsets,
                    subsettingcrs=req_crs,
                    outputcrs=req_crs
                )

        else:
            # WCS 1.0.0
            resp = wcs.getCoverage(
                identifier=coverage_id,
                bbox=bbox,
                crs=req_crs,
                format="GeoTIFF"
            )
        with open(tmp_path, "wb") as f:
            data = resp.read()
            f.write(data)

        # Clippen op polygon (in raster CRS)
        with rasterio.open(tmp_path) as src:
            raster_crs = src.crs.to_string() if src.crs else req_crs
            # Projecteer clip-geom naar raster CRS
            gdf_crs = getattr(clip_geom, "crs", None)
            # clip_geom is shapely; we kennen CRS via context, dus we transformeren expliciet
            if gdf_crs and gdf_crs != raster_crs:
                transformer = Transformer.from_crs(CRS.from_user_input(gdf_crs), CRS.from_user_input(raster_crs), always_xy=True)
                poly = shp_transform(transformer.transform, clip_geom)
            else:
                poly = clip_geom

            out_arr, out_transform = mask(src, [mapping(poly)], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_arr.shape[1],
                "width": out_arr.shape[2],
                "transform": out_transform
            })

        # Schrijf het eindresultaat
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

def pick_tiff_format(cov) -> str:
    fmts = []
    for attr in ("formats", "supportedFormats"):
        v = getattr(cov, attr, None)
        if v:
            fmts = list(v)
            break
    # voorkeursvolgorde
    for cand in fmts:
        s = cand.lower()
        if "geotiff" in s or "image/tiff" in s or "tiff" in s:
            return cand
    # laatste redmiddel
    return "image/tiff"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = read_targets()

    # Voorbereiden CRS: we vragen bbox in REQUEST_CRS
    gdf_req = gdf.to_crs(REQUEST_CRS)

    # Voor elke WCS: lijst coverages
    all_services = []
    for svc in WCS_SERVICES:
        alias = svc["alias"]
        url = svc["url"]
        filter_regex = svc.get("filter_regex")
        logging.info(f"Verbind met {alias}: {url}")
        wcs, version = get_wcs_client(url)
        covs = list_coverages(wcs, version, filter_regex)
        all_services.append({
            "alias": alias,
            "url": url,
            "wcs": wcs,
            "version": version,
            "coverages": covs
        })
        logging.info(f"{alias} ({version}): {len(covs)} coverages")

    # Itereer per feature
    for idx, row in gdf_req.iterrows():
        name_raw = str(row[NAME_COL])
        name = sanitize_filename(name_raw) or f"feat_{idx}"
        geom = row[GEOM_COL]
        if geom is None or geom.is_empty:
            logging.info(f"Rij {idx}: lege geometrie; sla over")
            continue

        # Bbox met optionele buffer (in REQUEST_CRS-eenheden)
        minx, miny, maxx, maxy = geom.bounds
        if BBOX_BUFFER > 0:
            minx -= BBOX_BUFFER
            miny -= BBOX_BUFFER
            maxx += BBOX_BUFFER
            maxy += BBOX_BUFFER
        bbox = (minx, miny, maxx, maxy)

        out_dir = OUTPUT_DIR / name
        ensure_dir(out_dir)

        logging.info(f"Feature '{name_raw}' -> {out_dir.name} | bbox {REQUEST_CRS}: {tuple(round(v,3) for v in bbox)}")

        # Voor elke service en coverage
        for svc in all_services:
            alias = svc["alias"]
            wcs = svc["wcs"]
            version = svc["version"]
            for cov in svc["coverages"]:
                cov_id = cov["identifier"]
                cov_title = cov["title"]
                yr = extract_year(cov_id) or extract_year(cov_title) or "unknown"

                # Bestandsnaam (botsing vermijden)
                base = f"{yr}.tif"
                out_path = out_dir / base
                if out_path.exists() and not OVERWRITE:
                    # voeg suffix toe
                    stem = out_path.stem
                    k = 2
                    candidate = out_dir / f"{stem}_{alias}.tif"
                    if not candidate.exists():
                        out_path = candidate
                    else:
                        while True:
                            candidate = out_dir / f"{stem}_{alias}_{k}.tif"
                            if not candidate.exists():
                                out_path = candidate
                                break
                            k += 1

                ok = wcs_get_and_clip(
                    wcs=wcs,
                    version=version,
                    coverage_id=cov_id,
                    req_crs=REQUEST_CRS,
                    bbox=bbox,
                    clip_geom=geom,   # geom is reeds in REQUEST_CRS dankzij gdf_req
                    out_path=out_path
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
