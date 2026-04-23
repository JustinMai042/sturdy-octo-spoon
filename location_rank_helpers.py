import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

NO_BUILDING_LABEL = "NO_BUILDING"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
NON_BUILDING_PROMPTS = [
    "a photo of trees",
    "a photo of a road",
    "a photo of the sky",
    "a landscape with no building",
]


@dataclass
class CandidateNode:
    name: str
    geometry: Any
    distance_m: float
    bearing_error_deg: float
    ray_hit: bool
    geometry_score: float
    vision_logit: float = 0.0
    probability: float = 0.0


def exif_gps_and_heading(photo_path: Path) -> tuple[float, float, float | None, float]:
    cmd = [
        "exiftool",
        "-j",
        "-n",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSHPositioningError",
        "-GPSImgDirection",
        "-GPSImgDirectionRef",
        str(photo_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "exiftool failed")

    meta = json.loads(result.stdout)[0] if result.stdout else {}
    if not meta:
        raise RuntimeError("exiftool returned empty JSON")

    for field in ("GPSLatitude", "GPSLongitude", "GPSImgDirection"):
        if field not in meta:
            raise RuntimeError(f"Missing {field} in EXIF")

    az_ref = meta.get("GPSImgDirectionRef", "T").strip().upper()
    if az_ref != "T":
        raise RuntimeError(f"GPSImgDirectionRef={az_ref!r} (not True). Need magnetic->true conversion.")

    return (
        float(meta["GPSLatitude"]),
        float(meta["GPSLongitude"]),
        float(meta["GPSHPositioningError"]) if "GPSHPositioningError" in meta else None,
        float(meta["GPSImgDirection"]),
    )


def auto_find_flightlog(photo_path: Path) -> Path | None:
    for parent in [photo_path.parent, *photo_path.parents]:
        candidate = parent / "flightlog.txt"
        if candidate.exists():
            return candidate
    return None


def candidate_relative_keys(photo_path: Path) -> list[str]:
    parts = list(photo_path.parts)
    keys: list[str] = []
    if "images" in parts:
        image_idx = parts.index("images")
        rel = "/".join(parts[image_idx + 1 :])
        if rel:
            keys.append(rel)
    if len(parts) >= 4:
        keys.append("/".join(parts[-4:]))
    return keys


@lru_cache(maxsize=8)
def load_flightlog_rows(flightlog_path: str) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for raw_line in Path(flightlog_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rel_path, lon, lat, altitude = [part.strip() for part in line.split(",", maxsplit=3)]
        rows.append((rel_path, float(lon), float(lat), float(altitude)))
    return rows


def resolve_flightlog_image_path(flightlog_path: Path, rel_path: str) -> Path | None:
    for candidate in (flightlog_path.parent / "images" / rel_path, flightlog_path.parent / rel_path):
        if candidate.exists():
            return candidate.resolve()
    return None


def lookup_flightlog_entry(photo_path: Path, flightlog_path: Path) -> tuple[str, float, float, float]:
    rows = load_flightlog_rows(str(flightlog_path))
    for key in candidate_relative_keys(photo_path):
        matches = [row for row in rows if row[0] == key or row[0].endswith(key)]
        if len(matches) == 1:
            return matches[0]
    raise RuntimeError(f"Could not find a unique flightlog entry for {photo_path.name} in {flightlog_path}")


def parse_xmp_sidecar(xmp_path: Path) -> tuple[list[float], list[float]]:
    ns = {"xcr": "http://www.capturingreality.com/ns/xcr/1.1#"}
    root = ET.parse(xmp_path).getroot()
    rotation_node = root.find(".//xcr:Rotation", ns)
    position_node = root.find(".//xcr:Position", ns)
    if rotation_node is None or position_node is None or not rotation_node.text or not position_node.text:
        raise RuntimeError(f"Missing rotation or position in {xmp_path}")

    rotation = [float(value) for value in rotation_node.text.split()]
    position = [float(value) for value in position_node.text.split()]
    if len(rotation) != 9 or len(position) != 3:
        raise RuntimeError(f"Unexpected XMP pose shape in {xmp_path}")
    return rotation, position


def lonlat_to_local_m(lon: float, lat: float, lon_ref: float, lat_ref: float) -> tuple[float, float]:
    earth_radius_m = 6378137.0
    east_m = math.radians(lon - lon_ref) * earth_radius_m * math.cos(math.radians(lat_ref))
    north_m = math.radians(lat - lat_ref) * earth_radius_m
    return east_m, north_m


@lru_cache(maxsize=8)
def flightlog_xmp_rotation_deg(flightlog_path: str) -> float:
    flightlog = Path(flightlog_path)
    rows = load_flightlog_rows(flightlog_path)
    if not rows:
        raise RuntimeError(f"Flightlog is empty: {flightlog}")

    lon_ref = sum(row[1] for row in rows) / len(rows)
    lat_ref = sum(row[2] for row in rows) / len(rows)
    samples: list[tuple[float, float, float, float]] = []

    for rel_path, lon, lat, _alt in rows:
        image_path = resolve_flightlog_image_path(flightlog, rel_path)
        if image_path is None:
            continue
        xmp_path = image_path.with_suffix(".xmp")
        if not xmp_path.exists():
            continue
        _, position = parse_xmp_sidecar(xmp_path)
        east_m, north_m = lonlat_to_local_m(lon, lat, lon_ref, lat_ref)
        samples.append((position[0], position[1], east_m, north_m))

    if len(samples) < 2:
        raise RuntimeError(f"Not enough flightlog/XMP pairs to estimate orientation from {flightlog}")

    local_x_center = sum(sample[0] for sample in samples) / len(samples)
    local_y_center = sum(sample[1] for sample in samples) / len(samples)
    geo_x_center = sum(sample[2] for sample in samples) / len(samples)
    geo_y_center = sum(sample[3] for sample in samples) / len(samples)

    cross = 0.0
    dot = 0.0
    for local_x, local_y, geo_x, geo_y in samples:
        ax = local_x - local_x_center
        ay = local_y - local_y_center
        bx = geo_x - geo_x_center
        by = geo_y - geo_y_center
        cross += ax * by - ay * bx
        dot += ax * bx + ay * by

    return math.degrees(math.atan2(cross, dot))


def flightlog_and_xmp_pose(
    photo_path: Path,
    flightlog_path: Path,
    heading_offset_deg: float = 0.0,
) -> tuple[float, float, float | None, float]:
    _rel_path, lon, lat, _alt = lookup_flightlog_entry(photo_path, flightlog_path)
    rotation, _position = parse_xmp_sidecar(photo_path.with_suffix(".xmp"))
    local_forward_x = rotation[6]
    local_forward_y = rotation[7]

    rotation_deg = flightlog_xmp_rotation_deg(str(flightlog_path))
    rotation_rad = math.radians(rotation_deg)
    east = math.cos(rotation_rad) * local_forward_x - math.sin(rotation_rad) * local_forward_y
    north = math.sin(rotation_rad) * local_forward_x + math.cos(rotation_rad) * local_forward_y
    heading_deg = (math.degrees(math.atan2(east, north)) + heading_offset_deg + 360.0) % 360.0
    return lat, lon, None, heading_deg


def photo_pose(
    photo_path: Path,
    flightlog_path: Path | None = None,
    heading_offset_deg: float = 0.0,
) -> tuple[float, float, float | None, float]:
    try:
        return exif_gps_and_heading(photo_path)
    except RuntimeError:
        pass

    candidate_flightlog = flightlog_path or auto_find_flightlog(photo_path)
    if candidate_flightlog is None:
        raise RuntimeError(
            f"No EXIF GPS/heading in {photo_path.name}, and no flightlog.txt was provided or found nearby."
        )
    return flightlog_and_xmp_pose(photo_path, candidate_flightlog, heading_offset_deg)


def normalize_angle_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def bearing_deg(origin: Point, target: Point) -> float:
    return (math.degrees(math.atan2(target.x - origin.x, target.y - origin.y)) + 360.0) % 360.0


def sector_polygon(origin: Point, heading_deg: float, radius_m: float, fov_deg: float, steps: int = 64) -> Polygon:
    start = heading_deg - fov_deg / 2.0
    angles = [math.radians(start + (fov_deg * i / steps)) for i in range(steps + 1)]
    arc = [(origin.x + math.sin(a) * radius_m, origin.y + math.cos(a) * radius_m) for a in angles]
    return Polygon([(origin.x, origin.y)] + arc)


def ray_entry_distance(origin: Point, ray: LineString, geom: Any) -> float | None:
    inter = ray.intersection(geom)
    if inter.is_empty:
        return None
    nearest_on_intersection, _ = nearest_points(inter, origin)
    return float(origin.distance(nearest_on_intersection))


def geometry_score(distance_m: float, bearing_error_deg: float, radius_m: float, fov_deg: float, ray_hit: bool) -> float:
    distance_term = max(0.0, 1.0 - distance_m / max(radius_m, 1.0))
    angle_term = max(0.0, 1.0 - bearing_error_deg / max(fov_deg / 2.0, 1.0))
    return 0.45 * distance_term + 0.45 * angle_term + (0.20 if ray_hit else 0.0)


def build_candidate_nodes(
    origin: Point,
    buildings: gpd.GeoDataFrame,
    name_col: str,
    heading_deg: float,
    radius_m: float,
    fov_deg: float,
    gps_acc_m: float | None,
    min_dist_m: float = 2.0,
    require_ray_hit: bool = False,
) -> tuple[list[CandidateNode], Polygon, LineString]:
    effective_radius = radius_m + (gps_acc_m or 0.0)
    cone = sector_polygon(origin, heading_deg, effective_radius, fov_deg)
    ray = LineString(
        [
            (origin.x, origin.y),
            (
                origin.x + math.sin(math.radians(heading_deg)) * effective_radius,
                origin.y + math.cos(math.radians(heading_deg)) * effective_radius,
            ),
        ]
    )

    idx = list(buildings.sindex.query(cone, predicate="intersects"))
    candidates = buildings.iloc[idx].copy() if idx else buildings.iloc[0:0].copy()
    best_by_name: dict[str, CandidateNode] = {}

    for _, row in candidates.iterrows():
        near, _ = nearest_points(row.geometry, origin)
        distance_m = float(origin.distance(near))
        if not (min_dist_m <= distance_m <= effective_radius):
            continue

        hit_dist = ray_entry_distance(origin, ray, row.geometry)
        ray_hit = hit_dist is not None
        if require_ray_hit and not ray_hit:
            continue

        node = CandidateNode(
            name=str(row[name_col]),
            geometry=row.geometry,
            distance_m=distance_m,
            bearing_error_deg=abs(normalize_angle_deg(bearing_deg(origin, near) - heading_deg)),
            ray_hit=ray_hit,
            geometry_score=geometry_score(
                distance_m,
                abs(normalize_angle_deg(bearing_deg(origin, near) - heading_deg)),
                effective_radius,
                fov_deg,
                ray_hit,
            ),
        )
        previous = best_by_name.get(node.name)
        if previous is None or (node.geometry_score, node.ray_hit, -node.distance_m) > (
            previous.geometry_score,
            previous.ray_hit,
            -previous.distance_m,
        ):
            best_by_name[node.name] = node

    return sorted(best_by_name.values(), key=lambda node: (-node.geometry_score, node.distance_m)), cone, ray


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def tokenize_name(value: str) -> list[str]:
    token = []
    tokens: list[str] = []
    for ch in value.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return tokens


def clean_reference_label(folder_name: str) -> str:
    label = folder_name.lower()
    for suffix in ("-undistorted_images", "_undistorted_images", "-undistorted-images"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    if "-upload" in label:
        label = label.split("-upload", 1)[0]
    while label and (label[0].isdigit() or label[0] in "-_ "):
        label = label[1:]
    return " ".join(tokenize_name(label)) or folder_name


def reference_match_score(candidate_name: str, reference_label: str) -> float:
    candidate_norm = normalize_name(candidate_name)
    reference_norm = normalize_name(reference_label)
    if not candidate_norm or not reference_norm:
        return 0.0
    if candidate_norm == reference_norm:
        return 10.0
    if candidate_norm in reference_norm or reference_norm in candidate_norm:
        return 8.0

    overlap = set(tokenize_name(candidate_name)) & set(tokenize_name(reference_label))
    if overlap:
        return 4.0 + len(overlap) / max(len(tokenize_name(candidate_name)), len(tokenize_name(reference_label)), 1)
    return 0.0


def discover_reference_folders(reference_root: Path) -> dict[str, list[Path]]:
    if not reference_root.exists():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")

    child_dataset_dirs = sorted(path for path in reference_root.iterdir() if path.is_dir() and (path / "images").is_dir())
    dataset_dirs = child_dataset_dirs or ([reference_root] if (reference_root / "images").is_dir() else [])
    reference_groups: dict[str, list[Path]] = {}

    for dataset_dir in dataset_dirs:
        label = clean_reference_label(dataset_dir.name)
        image_paths = sorted(
            path
            for path in (dataset_dir / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_paths:
            reference_groups.setdefault(label, []).extend(image_paths)

    return reference_groups


def reference_images_for_name(name: str, reference_catalog: dict[str, list[Path]]) -> list[Path]:
    if not reference_catalog:
        return []
    if name in reference_catalog:
        return reference_catalog[name]

    best_score, best_label = max(
        ((reference_match_score(name, reference_label), reference_label) for reference_label in reference_catalog),
        default=(0.0, ""),
    )
    return reference_catalog[best_label] if best_score > 0.0 else []


def score_nodes_with_clip(
    photo_path: Path,
    nodes: list[CandidateNode],
    reference_root: Path | None = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> tuple[list[CandidateNode], float, bool]:
    if not nodes:
        return [], 0.0, False

    try:
        from PIL import Image
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return nodes, 0.0, False

    reference_catalog = discover_reference_folders(reference_root) if reference_root else {}
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    with Image.open(photo_path) as image:
        query_image = image.convert("RGB")

    with torch.no_grad():
        q_inputs = {k: v.to(device) for k, v in processor(images=query_image, return_tensors="pt").items()}
        query_features = model.get_image_features(**q_inputs)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        non_building_inputs = {
            k: v.to(device) for k, v in processor(text=NON_BUILDING_PROMPTS, return_tensors="pt", padding=True).items()
        }
        non_building_features = model.get_text_features(**non_building_inputs)
        non_building_features = non_building_features / non_building_features.norm(dim=-1, keepdim=True)
        non_building_logit = float(torch.matmul(query_features, non_building_features.T).squeeze(0).max())

    for node in nodes:
        image_paths = reference_images_for_name(node.name, reference_catalog)
        with torch.no_grad():
            if image_paths:
                ref_images = []
                for image_path in image_paths:
                    with Image.open(image_path) as ref_image:
                        ref_images.append(ref_image.convert("RGB"))
                ref_inputs = {
                    k: v.to(device) for k, v in processor(images=ref_images, return_tensors="pt", padding=True).items()
                }
                ref_features = model.get_image_features(**ref_inputs)
                ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                mean_ref = ref_features.mean(0, keepdim=True)
                mean_ref = mean_ref / mean_ref.norm(dim=-1, keepdim=True)
                node.vision_logit = float(torch.matmul(query_features, mean_ref.T).squeeze())
            else:
                prompts = [
                    f"a photo of {node.name}",
                    f"an outdoor campus photo of {node.name}",
                    f"a university building named {node.name}",
                ]
                text_inputs = {k: v.to(device) for k, v in processor(text=prompts, return_tensors="pt", padding=True).items()}
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                node.vision_logit = float(torch.matmul(query_features, text_features.T).squeeze(0).max())

    return nodes, non_building_logit, True


def rank_nodes(nodes: list[CandidateNode], geometry_weight: float, vision_weight: float) -> list[CandidateNode]:
    if not nodes:
        return []
    logits = [
        geometry_weight * node.geometry_score + vision_weight * node.vision_logit + (0.05 if node.ray_hit else 0.0)
        for node in nodes
    ]
    for node, probability in zip(nodes, softmax(logits)):
        node.probability = probability
    return sorted(nodes, key=lambda node: node.probability, reverse=True)


def decide_result(
    ranked_nodes: list[CandidateNode],
    non_building_logit: float,
    min_top_prob: float,
    min_margin: float,
    min_vision_logit: float,
    non_building_threshold: float,
    use_vision: bool,
) -> tuple[str, list[str]]:
    if not ranked_nodes:
        return NO_BUILDING_LABEL, ["no_candidates_in_cone"]

    top = ranked_nodes[0]
    second_prob = ranked_nodes[1].probability if len(ranked_nodes) > 1 else 0.0
    reasons: list[str] = []
    if top.probability < min_top_prob:
        reasons.append("low_top_probability")
    if top.probability - second_prob < min_margin:
        reasons.append("small_top_margin")
    if use_vision and top.vision_logit < min_vision_logit:
        reasons.append("low_vision_logit")
    if use_vision and non_building_logit >= non_building_threshold:
        reasons.append("non_building_signal")
    return (NO_BUILDING_LABEL, reasons) if reasons else (top.name, ["confident_building_match"])


def print_result(
    photo_path: Path,
    lat: float,
    lon: float,
    acc_m: float | None,
    az: float,
    ranked_nodes: list[CandidateNode],
    predicted_label: str,
    reasons: list[str],
    non_building_logit: float,
    vision_available: bool,
) -> None:
    print(f"PHOTO: {photo_path.name}")
    print(f"WGS84: {lat:.7f}, {lon:.7f}")
    if acc_m is not None:
        print(f"EXIF accuracy (m): {acc_m:.1f}")
    print(f"AZIMUTH deg (true): {az:.1f}")

    print("RANKED PREDICTIONS:")
    if not ranked_nodes:
        print(f"  - {NO_BUILDING_LABEL}")
    for node in ranked_nodes[:5]:
        line = (
            f"  - {node.name}: prob={node.probability:.1%}, "
            f"dist={node.distance_m:.1f}m, bearing_err={node.bearing_error_deg:.1f}deg"
        )
        if vision_available:
            line += f", vision={node.vision_logit:.3f}"
        print(line)

    print("\nRESULT:")
    print(f"  label={predicted_label}")
    print(f"  reasons={', '.join(reasons)}")
    if ranked_nodes:
        print(f"  top_probability={ranked_nodes[0].probability:.1%}")
    if vision_available:
        print(f"  non_building_logit={non_building_logit:.3f}")


def export_geojson(
    output_path: Path,
    work_crs: str,
    photo_name: str,
    origin: Point,
    cone: Polygon,
    ray: LineString,
    ranked_nodes: list[CandidateNode],
    predicted_label: str,
) -> None:
    rows: list[dict[str, Any]] = [
        {"type": "origin", "name": photo_name, "score": 1.0, "geometry": origin},
        {"type": "view_cone", "name": "view_cone", "score": 1.0, "geometry": cone},
        {"type": "ray", "name": "camera_ray", "score": 1.0, "geometry": ray},
        {"type": "result", "name": predicted_label, "score": ranked_nodes[0].probability if ranked_nodes else 0.0, "geometry": origin},
    ]
    for node in ranked_nodes[:5]:
        rows.append(
            {
                "type": "candidate",
                "name": node.name,
                "score": node.probability,
                "geometry": node.geometry.boundary,
            }
        )
    gpd.GeoDataFrame(rows, crs=work_crs).to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
