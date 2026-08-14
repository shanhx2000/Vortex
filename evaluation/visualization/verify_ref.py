"""Check that the plot code still reproduces the committed reference figures.

Renders every figure from `stats/ref/` and writes a side-by-side sheet
(committed figures/ref/ | fresh render) for each, so the whole set can be
checked in one pass.

    python verify_ref.py                 # render + build comparison sheets
    python verify_ref.py --only BA BD
    python verify_ref.py --contact-sheet # also stack them into one image

**Every sheet is expected to match.** `stats/ref/` and `figures/ref/` are one
consistent set: the figures were rendered from that data by this code. A
mismatch here means the plot code changed, not that the numbers moved.

To see whether *your* numbers moved, render your own run instead and compare
that against figures/ref/:

    python plot_all.py --use-simulation --use-algorithm    # -> figures/

WHY THERE IS NO PASS/FAIL PIXEL TEST
------------------------------------
There was one, and it did not work. Different matplotlib versions compute
slightly different `tight_layout` margins from their own font metrics, so a
figure with byte-identical content renders with its axes shifted or stretched by
a few pixels. Measured on this set: `kernel_benchmark.pdf` is a confirmed visual
match yet scores 10.7% mean pixel difference, while two *genuinely different*
figures score 5.8%. The metric has no discriminating power, and a threshold
would either pass everything or cry wolf.

So the pixel numbers below are printed as advisory only. What the scripts print
to stdout -- geomeans, per-kernel speedups, area and power totals -- is the
reliable check, because those are the numbers the figure is made of.

Requires pdftoppm (poppler-utils) and Pillow to build the sheets. Without them
the figures are still rendered; only the comparison images are skipped.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import _paths
from plot_all import FIGURES

PAD = 24


def rasterize(pdf_path, out_prefix, dpi):
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    str(pdf_path), str(out_prefix)],
                   check=True, capture_output=True)
    return Path(f"{out_prefix}.png")


def side_by_side(ref_png, new_png, out_png):
    from PIL import Image, ImageChops, ImageDraw

    a = Image.open(ref_png).convert("RGB")
    b = Image.open(new_png).convert("RGB")
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width + PAD * 3, h + PAD * 3), "white")
    canvas.paste(a, (PAD, PAD * 2))
    canvas.paste(b, (a.width + PAD * 2, PAD * 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, PAD // 2), "committed figures/ref/", fill="black")
    draw.text((a.width + PAD * 2, PAD // 2), "fresh render from stats/ref/", fill="black")
    canvas.save(out_png)

    # Advisory only -- see the module docstring.
    w, hh = max(a.width, b.width), h
    pa = Image.new("RGB", (w, hh), "white"); pa.paste(a, (0, 0))
    pb = Image.new("RGB", (w, hh), "white"); pb.paste(b, (0, 0))
    diff = ImageChops.difference(pa, pb).convert("L")
    hist = diff.histogram()
    differing = sum(hist[17:])
    return differing / float(w * hh), (a.size, b.size)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="INDEX")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--contact-sheet", action="store_true",
                        help="also stack every sheet into one tall image")
    args = parser.parse_args()

    out_dir = _paths.apply_common_args(argparse.Namespace(out_dir=None))

    # Resolved against this file, not the CWD, so the script works from the
    # repository root as the README invokes it.
    plot_all = Path(__file__).resolve().parent / "plot_all.py"
    cmd = [sys.executable, str(plot_all), "--use-ref"]  # explicit; also the default
    if args.only:
        cmd += ["--only"] + args.only
    if subprocess.call(cmd) != 0:
        print("\nrendering failed; not comparing", file=sys.stderr)
        return 1

    have_tools = shutil.which("pdftoppm") is not None
    try:
        import PIL  # noqa: F401
    except ImportError:
        have_tools = False
    if not have_tools:
        print("\npdftoppm and/or Pillow unavailable -- rendered but no sheets built.")
        print(f"Compare by hand: {out_dir} vs {_paths.REF_FIGURES_DIR}")
        print(f"  (submitted-era renders, for reference: {_paths.INITIAL_SUBMISSION_DIR})")
        return 0

    sheet_dir = out_dir / "comparison"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    raster_dir = sheet_dir / ".raster"
    raster_dir.mkdir(exist_ok=True)

    figures = [f for f in FIGURES if not args.only or f[0] in args.only]
    seen, rows, sheets = set(), [], []

    for index, module_name, _, kwargs in figures:
        module = __import__(module_name)
        if kwargs.get("workload"):
            in_len, out_len = module.WORKLOADS[kwargs["workload"]][:2]
            name = f"vortex_hardware_speedup_inlen{in_len}_outlen{out_len}.pdf"
        else:
            name = module.FIGURE
        if name in seen:
            continue
        seen.add(name)

        new_pdf, ref_pdf = out_dir / name, _paths.reference_figure(name)
        if not new_pdf.exists():
            rows.append((index, name, None, "NOT RENDERED"))
            continue
        if not ref_pdf.exists():
            rows.append((index, name, None, "no reference on file"))
            continue

        sheet = sheet_dir / f"{index}_{name.replace('.pdf', '')}.png"
        ratio, sizes = side_by_side(
            rasterize(ref_pdf, raster_dir / f"{index}_ref", args.dpi),
            rasterize(new_pdf, raster_dir / f"{index}_new", args.dpi),
            sheet)
        sheets.append(sheet)
        note = "" if sizes[0] == sizes[1] else f"  (sizes {sizes[0]} vs {sizes[1]})"
        rows.append((index, name, ratio, f"sheet written{note}"))

    print(f"\n{'fig':<5}{'file':<52}{'pixel diff':>11}  note")
    print("-" * 92)
    for index, name, ratio, note in rows:
        shown = f"{ratio * 100:9.2f}%" if ratio is not None else "        -"
        print(f"{index:<5}{name:<52}{shown}  {note}")
    print("-" * 92)

    shutil.rmtree(raster_dir, ignore_errors=True)

    if args.contact_sheet and sheets:
        from PIL import Image
        ims = [Image.open(s) for s in sorted(sheets)]
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            canvas.paste(i, (0, y))
            y += i.height
        combined = sheet_dir / "all_figures.png"
        canvas.save(combined)
        print(f"contact sheet: {combined}")

    missing = [r for r in rows if r[2] is None]
    print(f"\n{len(rows) - len(missing)} sheet(s) in {sheet_dir}")
    print("Pixel diff is ADVISORY -- cross-version layout shifts inflate it.")
    print("Look at the sheets, and at the aggregates each script prints.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
