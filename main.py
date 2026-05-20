"""
main.py - CLI entry point for the 3D-to-Technical Export tool.

Usage:
    python main.py sample_scene.json --format pdf --views top front iso --units meters
    python main.py sample_scene.json --format dxf --views top --units feet
    python main.py sample_scene.json --format both
"""

import argparse
import os
import sys

from scene_parser import parse_scene
from renderer_pdf import PDFRenderer
from renderer_dxf import DXFRenderer


def main():
    parser = argparse.ArgumentParser(
        description="3D-to-Technical Export: Convert 3D scene JSON to professional technical PDF/DXF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py sample_scene.json
  python main.py sample_scene.json --format pdf --views top front iso
  python main.py sample_scene.json --format dxf --views top --units feet
  python main.py sample_scene.json --format both --output output/drawing
        """,
    )

    parser.add_argument(
        "scene",
        help="Path to the 3D scene JSON file",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["pdf", "dxf", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--views", "-v",
        nargs="+",
        choices=["top", "front", "iso"],
        default=["top", "front", "iso"],
        help="Views to render (default: top front iso)",
    )
    parser.add_argument(
        "--units", "-u",
        choices=["meters", "feet"],
        default="meters",
        help="Dimension units (default: meters)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (without extension). Default: based on input filename",
    )
    parser.add_argument(
        "--page-size", "-p",
        choices=["A3", "A4"],
        default="A3",
        help="PDF page size (default: A3)",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.scene):
        print(f"Error: Scene file not found: {args.scene}", file=sys.stderr)
        sys.exit(1)

    # Parse scene — auto-detect format by extension
    ext = os.path.splitext(args.scene)[1].lower()
    print(f"Parsing scene: {args.scene}")
    if ext in (".glb", ".gltf"):
        from glb_parser import scene_from_glb
        scene = scene_from_glb(args.scene, units=args.units)
    else:
        scene = parse_scene(args.scene)
    print(f"  Title: {scene.metadata.title}")
    print(f"  Walls: {len(scene.walls)}")
    print(f"  Floors: {len(scene.floors)}")
    print(f"  Doors: {len(scene.doors)}")
    print(f"  Track Spotlights: {len(scene.track_spotlights)}")
    print(f"  LED Strips: {len(scene.led_strips)}")
    print(f"  Truss Segments: {len(scene.truss_segments)}")
    print(f"  Furniture: {len(scene.furniture)}")
    print(f"  Assets: {len(scene.assets)}")

    # Determine output path
    if args.output:
        base_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.scene))[0]
        base_path = os.path.join(os.path.dirname(args.scene) or ".", f"{base_name}_technical")

    # Ensure output directory exists
    out_dir = os.path.dirname(base_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Render PDF
    if args.format in ("pdf", "both"):
        from reportlab.lib.pagesizes import A3, A4, landscape
        page_size = landscape(A3) if args.page_size == "A3" else landscape(A4)

        pdf_path = f"{base_path}.pdf"
        print(f"\nRendering PDF: {pdf_path}")
        print(f"  Views: {', '.join(args.views)}")
        print(f"  Units: {args.units}")
        print(f"  Page Size: {args.page_size} Landscape")

        renderer = PDFRenderer(pdf_path, page_size=page_size, units=args.units)
        renderer.render(scene, views=args.views)
        print(f"  PDF saved successfully ({len(args.views)} page(s))")
        print(f"  Layers: 'Layer 1 - Architecture', 'Layer 2 - Lighting & Rigging', 'Layer 3 - Annotations'")

    # Render DXF
    if args.format in ("dxf", "both"):
        dxf_path = f"{base_path}.dxf"
        print(f"\nRendering DXF: {dxf_path}")
        print(f"  Views: {', '.join(args.views)}")
        print(f"  Units: {args.units}")

        renderer = DXFRenderer(dxf_path, units=args.units)
        renderer.render(scene, views=args.views)
        print(f"  DXF saved successfully")
        print(f"  Layers: '1-ARCHITECTURE', '2-LIGHTING_RIGGING', '3-ANNOTATIONS'")

    print("\nDone!")


if __name__ == "__main__":
    main()
