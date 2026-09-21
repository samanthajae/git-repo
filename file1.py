import re
import pandas as pd
import numpy as np
from dataclasses import dataclass
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import MiniMap
from IPython.display import display

#define function to convert PSA grid coordinates to latitude & longitude based on rules

MINI_GRID_OFFSETS = {
    "A": (+0.3, +0.3),
    "B": (+0.7, +0.3),
    "C": (-0.3, -0.3),
    "D": (+0.7, -0.3),
}


@dataclass
class GridResult:
    grid: str
    lat_dms: str
    lon_dms: str
    latitude: float
    longitude: float

    def __str__(self):
        return (f"{self.grid}: {self.lat_dms} / {self.lon_dms}  ->  "
                f"({self.latitude:.6f}, {self.longitude:.6f})")


def _dms(deg, minutes, hemi):
    whole = int(minutes)
    seconds = round((minutes - whole) * 60, 1)
    return f"{deg}\u00b0{hemi} {whole:02d}' {seconds:04.1f}\""


def convert(grid):
    g = grid.strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d{2})(\d{2})([ABCD])?", g)
    if not m:
        raise ValueError(f"Invalid grid coordinate: {grid!r} (expected e.g. 5107A)")
    east_min, north_min, mini = int(m.group(1)), int(m.group(2)), m.group(3)

    if east_min >= 60 or north_min >= 60:
        raise ValueError(f"Minutes must be 0-59 in {grid!r}")

    #rules 1, 2: first two digits -> E/W axis minutes.
    #leading digit 0 rolls over into the next degree (104), otherwise 103.
    lon_deg = 104 if m.group(1)[0] == "0" else 103
    lon_min = float(east_min)

    #rule 3: last two digits -> N/S axis minutes, base degree 1.
    lat_deg = 1
    lat_min = float(north_min)

    if mini:
        d_lon, d_lat = MINI_GRID_OFFSETS[mini]
        lon_min += d_lon
        lat_min += d_lat

    lon_deg, lon_min = lon_deg + int(lon_min // 60), lon_min % 60
    lat_deg, lat_min = lat_deg + int(lat_min // 60), lat_min % 60

    return GridResult(
        grid=g,
        lat_dms=_dms(lat_deg, lat_min, "N"),
        lon_dms=_dms(lon_deg, lon_min, "E"),
        latitude=round(lat_deg + lat_min / 60, 6),
        longitude=round(lon_deg + lon_min / 60, 6),
    )

def convert_many(grids):
    return [convert(g) for g in grids]


#define function to convert PSA grid coordinates to latitude & longitude in df

def add_latlon_columns(df, grid_col="B", lat_col="Latitude", lon_col="Longitude",
                       errors="coerce", inplace=False, insert_after_grid=True):
    out = df if inplace else df.copy()

    if isinstance(grid_col, int):
        pos = grid_col
    elif grid_col in out.columns:
        pos = out.columns.get_loc(grid_col)
    elif out.shape[1] >= 2:
        pos = 1  #fall back to column B by position
    else:
        raise KeyError(f"Cannot locate grid column {grid_col!r}")

    lats, lons = [], []
    for value in out.iloc[:, pos]:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
                raise ValueError("empty grid value")
            r = convert(str(value))
            lats.append(r.latitude)
            lons.append(r.longitude)
        except Exception:
            if errors == "raise":
                raise
            lats.append(np.nan)
            lons.append(np.nan)

    for col in (lat_col, lon_col):
        if col in out.columns:
            out.drop(columns=col, inplace=True)

    at = pos + 1 if insert_after_grid else out.shape[1]
    out.insert(min(at, out.shape[1]), lat_col, lats)
    out.insert(min(at + 1, out.shape[1]), lon_col, lons)
    return out

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

DOT_COLOR = "#d62728" 
DOT_SIZE = 8

def show_map(lat, lon, label=None, zoom=13, tiles="OpenStreetMap", html_path=None,
             dot_color=DOT_COLOR, dot_size=DOT_SIZE, fill_opacity=0.9, outline="white"):
    """Return an interactive Folium map centred on (lat, lon) with a coloured dot."""
    _validate(lat, lon)
    fmap = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles, control_scale=True)
    folium.CircleMarker(
        [lat, lon],
        radius=dot_size,
        color=outline,
        weight=2,
        fill=True,
        fill_color=dot_color,
        fill_opacity=fill_opacity,
        tooltip=label or f"{lat:.6f}, {lon:.6f}",
        popup=folium.Popup(f"<b>{label or 'Location'}</b><br>Lat: {lat}<br>Lon: {lon}", max_width=250),
    ).add_to(fmap)
    MiniMap(toggle_display=True).add_to(fmap)
    if html_path:
        fmap.save(html_path)
        print(f"Interactive map saved to {html_path}")
    return fmap


def show_points(points, zoom=None, tiles="OpenStreetMap", html_path=None,
                dot_color=DOT_COLOR, dot_size=DOT_SIZE, fill_opacity=0.9, outline="white"):
    """points = [(lat, lon, label), ...] -> interactive map with all pins."""
    pts = [(p[0], p[1], p[2] if len(p) > 2 else None) for p in points]
    for lat, lon, _ in pts:
        _validate(lat, lon)
    center = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
    fmap = folium.Map(location=center, zoom_start=zoom or 12, tiles=tiles, control_scale=True)
    for lat, lon, label in pts:
        folium.CircleMarker(
            [lat, lon],
            radius=dot_size,
            color=outline,
            weight=2,
            fill=True,
            fill_color=dot_color,
            fill_opacity=fill_opacity,
            tooltip=label or f"{lat:.6f}, {lon:.6f}",
        ).add_to(fmap)
    if len(pts) > 1 and zoom is None:
        fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                         [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(30, 30))
    if html_path:
        fmap.save(html_path)
        print(f"Interactive map saved to {html_path}")
    return fmap


def save_map_image(lat, lon, path="map.png", zoom=13, width=1000, height=750,
                   dot_color=DOT_COLOR, dot_size=DOT_SIZE, outline="white",
                   quality=92, extra_points=None):
    """Render a static map image with a coloured dot and save as .png / .jpg / .jpeg."""
    _validate(lat, lon)
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("path must end in .png, .jpg or .jpeg")

    smap = StaticMap(width, height, url_template=TILE_URL,
                     headers={"User-Agent": "coordinate-map-notebook/1.0"})
    for plat, plon in [(lat, lon)] + list(extra_points or []):
        if outline:
            smap.add_marker(CircleMarker((plon, plat), outline, dot_size * 2 + 4))  # outline
        smap.add_marker(CircleMarker((plon, plat), dot_color, dot_size * 2))        # dot
    image = smap.render(zoom=zoom)

    if ext == ".png":
        image.save(path)
    else:
        image.convert("RGB").save(path, "JPEG", quality=quality)
    print(f"Image saved to {os.path.abspath(path)}")
    return path


def _validate(lat, lon):
    if not (-90 <= float(lat) <= 90):
        raise ValueError(f"Latitude out of range: {lat}")
    if not (-180 <= float(lon) <= 180):
        raise ValueError(f"Longitude out of range: {lon}")

def ask_and_map():
    label = input("Label (optional): ").strip() or None
    zoom = int(input("Zoom 1-19 [13]: ") or 13)
    #color = input(f"Dot colour [{DOT_COLOR}]: ").strip() or DOT_COLOR
    #size = int(input(f"Dot size (px radius) [{DOT_SIZE}]: ") or DOT_SIZE)
    save_map_image(lat, lon, path="map.png", zoom=zoom) #dot_color=color, dot_size=size
    return show_map(lat, lon, label=label, zoom=zoom, html_path="map.html")
                    #dot_color=color, #dot_size=size