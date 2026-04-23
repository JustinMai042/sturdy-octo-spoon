import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from location_rank_helpers import (
    build_candidate_nodes,
    decide_result,
    export_geojson,
    photo_pose,
    print_result,
    rank_nodes,
    score_nodes_with_clip,
)


def main(
    photo: str,
    buildings_gpkg: str,
    buildings_layer: str,
    name_col: str = "Name",
    candidate_radius_m: float = 300.0,
    work_crs: str = "EPSG:32610",
    fov_deg: float = 70.0,
    reference_root: str | None = None,
    geometry_weight: float = 0.35,
    vision_weight: float = 0.65,
    min_top_prob: float = 0.45,
    min_margin: float = 0.12,
    min_vision_logit: float = 0.20,
    non_building_threshold: float = 0.24,
    min_dist_m: float = 2.0,
    require_ray_hit: bool = False,
    output_geojson: str = "photo_look_result.geojson",
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    flightlog: str | None = None,
    heading_offset_deg: float = 0.0,
) -> str:
    photo_path = Path(photo).expanduser().resolve()
    buildings_path = Path(buildings_gpkg).expanduser().resolve()
    reference_root_path = Path(reference_root).expanduser().resolve() if reference_root else None
    output_path = Path(output_geojson).expanduser().resolve()
    flightlog_path = Path(flightlog).expanduser().resolve() if flightlog else None

    if not photo_path.exists():
        raise FileNotFoundError(f"Photo not found: {photo_path}")
    if not buildings_path.exists():
        raise FileNotFoundError(f"Buildings gpkg not found: {buildings_path}")

    lat, lon, acc_m, az = photo_pose(photo_path, flightlog_path, heading_offset_deg)
    origin = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(work_crs).iloc[0]

    buildings = gpd.read_file(buildings_path, layer=buildings_layer).to_crs(work_crs)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if buildings.empty:
        raise RuntimeError("No polygon geometry in building layer")
    if name_col not in buildings.columns:
        raise RuntimeError(f"name_col {name_col!r} not found. Available: {list(buildings.columns)}")

    nodes, cone, ray = build_candidate_nodes(
        origin=origin,
        buildings=buildings,
        name_col=name_col,
        heading_deg=az,
        radius_m=candidate_radius_m,
        fov_deg=fov_deg,
        gps_acc_m=acc_m,
        min_dist_m=min_dist_m,
        require_ray_hit=require_ray_hit,
    )

    non_building_logit = 0.0
    vision_available = False
    if nodes:
        try:
            nodes, non_building_logit, vision_available = score_nodes_with_clip(
                photo_path=photo_path,
                nodes=nodes,
                reference_root=reference_root_path,
                model_name=model_name,
                device=device,
            )
        except Exception as exc:
            print(f"Vision scoring unavailable, falling back to geometry only: {exc}")

    ranked_nodes = rank_nodes(nodes, geometry_weight, vision_weight if vision_available else 0.0)
    predicted_label, reasons = decide_result(
        ranked_nodes=ranked_nodes,
        non_building_logit=non_building_logit,
        min_top_prob=min_top_prob,
        min_margin=min_margin,
        min_vision_logit=min_vision_logit,
        non_building_threshold=non_building_threshold,
        use_vision=vision_available,
    )

    print_result(
        photo_path=photo_path,
        lat=lat,
        lon=lon,
        acc_m=acc_m,
        az=az,
        ranked_nodes=ranked_nodes,
        predicted_label=predicted_label,
        reasons=reasons,
        non_building_logit=non_building_logit,
        vision_available=vision_available,
    )
    export_geojson(output_path, work_crs, photo_path.name, origin, cone, ray, ranked_nodes, predicted_label)
    return predicted_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank likely campus buildings from a geotagged photo.")
    parser.add_argument("photo")
    parser.add_argument("buildings_gpkg")
    parser.add_argument("buildings_layer")
    parser.add_argument("name_col", nargs="?", default="Name")
    parser.add_argument("candidate_radius_m", nargs="?", type=float, default=300.0)
    parser.add_argument("--work-crs", default="EPSG:32610")
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--geometry-weight", type=float, default=0.35)
    parser.add_argument("--vision-weight", type=float, default=0.65)
    parser.add_argument("--min-top-prob", type=float, default=0.45)
    parser.add_argument("--min-margin", type=float, default=0.12)
    parser.add_argument("--min-vision-logit", type=float, default=0.20)
    parser.add_argument("--non-building-threshold", type=float, default=0.24)
    parser.add_argument("--min-dist-m", type=float, default=2.0)
    parser.add_argument("--require-ray-hit", action="store_true")
    parser.add_argument("--output-geojson", default="photo_look_result.geojson")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flightlog", default=None)
    parser.add_argument("--heading-offset-deg", type=float, default=0.0)
    args = parser.parse_args()

    main(
        photo=args.photo,
        buildings_gpkg=args.buildings_gpkg,
        buildings_layer=args.buildings_layer,
        name_col=args.name_col,
        candidate_radius_m=args.candidate_radius_m,
        work_crs=args.work_crs,
        fov_deg=args.fov_deg,
        reference_root=args.reference_root,
        geometry_weight=args.geometry_weight,
        vision_weight=args.vision_weight,
        min_top_prob=args.min_top_prob,
        min_margin=args.min_margin,
        min_vision_logit=args.min_vision_logit,
        non_building_threshold=args.non_building_threshold,
        min_dist_m=args.min_dist_m,
        require_ray_hit=args.require_ray_hit,
        output_geojson=args.output_geojson,
        model_name=args.model_name,
        device=args.device,
        flightlog=args.flightlog,
        heading_offset_deg=args.heading_offset_deg,
    )
