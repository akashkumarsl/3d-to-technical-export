"""Validate generated PDF and DXF outputs."""
import os
import sys

# --- PDF validation ---
def validate_pdf(path):
    print(f"\nValidating PDF: {path}")
    if not os.path.exists(path):
        print(f"  FAIL: file does not exist")
        return False
    size = os.path.getsize(path)
    print(f"  Size: {size:,} bytes")
    with open(path, "rb") as f:
        head = f.read(8)
        if not head.startswith(b"%PDF-"):
            print(f"  FAIL: missing PDF header")
            return False
        print(f"  Header: {head.decode(errors='ignore').strip()}")
        f.seek(-1024, 2)
        tail = f.read()
        if b"%%EOF" not in tail:
            print(f"  FAIL: missing %%EOF marker")
            return False
        print(f"  EOF: present")
    # Check OCG layer presence
    with open(path, "rb") as f:
        content = f.read()
    layers_found = []
    for layer_name in [b"Layer 1 - Architecture",
                       b"Layer 2 - Lighting & Rigging",
                       b"Layer 3 - Annotations"]:
        if layer_name in content:
            layers_found.append(layer_name.decode())
    print(f"  Layers detected: {len(layers_found)}/3 — {layers_found}")
    if b"/OCProperties" in content:
        print(f"  OCProperties: present (PDF viewers will show layer panel)")
    else:
        print(f"  WARNING: no OCProperties block")
    # Count pages by looking for /Type /Page entries
    page_count = content.count(b"/Type /Page\n") + content.count(b"/Type/Page\n")
    print(f"  Page entries (approx): {page_count}")
    return True


# --- DXF validation ---
def validate_dxf(path):
    print(f"\nValidating DXF: {path}")
    if not os.path.exists(path):
        print(f"  FAIL: file does not exist")
        return False
    try:
        import ezdxf
        doc = ezdxf.readfile(path)
        print(f"  DXF version: {doc.dxfversion}")
        layers = [l.dxf.name for l in doc.layers
                  if not l.dxf.name.startswith(("0", "Defpoints"))]
        print(f"  User layers: {layers}")
        msp = doc.modelspace()
        entity_counts = {}
        for e in msp:
            entity_counts[e.dxftype()] = entity_counts.get(e.dxftype(), 0) + 1
        print(f"  Modelspace entities: {dict(entity_counts)}")
        total = sum(entity_counts.values())
        print(f"  Total entities: {total}")
        # Count entities per layer
        layer_counts = {}
        for e in msp:
            ln = e.dxf.layer
            layer_counts[ln] = layer_counts.get(ln, 0) + 1
        print(f"  Entities per layer: {dict(layer_counts)}")
        return True
    except Exception as ex:
        print(f"  FAIL: {ex}")
        return False


if __name__ == "__main__":
    files = [
        "booth_technical.pdf",
        "booth_technical.dxf",
        "sample_scene_technical.pdf",
        "sample_scene_technical.dxf",
    ]
    all_ok = True
    for f in files:
        if f.endswith(".pdf"):
            ok = validate_pdf(f)
        else:
            ok = validate_dxf(f)
        if not ok:
            all_ok = False
    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED" if all_ok else "SOME VALIDATIONS FAILED")
    sys.exit(0 if all_ok else 1)
