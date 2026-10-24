"""Build small mirrored asphalt/rocks boards without starting Isaac Sim."""
import argparse
from pathlib import Path

from create_binary_choice_asphalt_rocks import generate, SOURCE
from pxr import Usd, UsdGeom, Gf, Vt


def taper_rocks(path, width, flat_width=0.0):
    """Bring the actual cropped 2x2 m surface perimeter to zero, keeping skirts."""
    stage = Usd.Stage.Open(str(path))
    for cell in stage.GetPrimAtPath('/World/Grid').GetChildren():
        if not cell.GetName().endswith('t5_large_rocks'):
            continue
        _, row, col, _ = cell.GetName().split('_', 3)
        cx, cy = 2 * int(col), 2 * int(row)
        for prim in Usd.PrimRange(cell):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = []
            boundary_count = 0
            for p in mesh.GetPointsAttr().Get():
                x, y, z = p
                if z >= -0.0001:
                    dx, dy = max(0., 1 - abs(x-cx)), max(0., 1 - abs(y-cy))
                    u, v = min(1., max(0., dx-flat_width)/width), min(1., max(0., dy-flat_width)/width)
                    z = max(0., z) * (u*u*(3-2*u)) * (v*v*(3-2*v))
                    if min(dx, dy) < 0.0001:
                        z = 0.
                        boundary_count += 1
                points.append(Gf.Vec3f(x, y, z))
            if not boundary_count:
                raise RuntimeError(f'No surface perimeter found: {prim.GetPath()}')
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--rocks-edge-width', type=float, default=0.0,
                        help='Smooth transition to zero at each cropped rock border, in metres.')
    parser.add_argument('--rocks-flat-border', type=float, default=0.0,
                        help='Exactly flat strip before the edge transition, in metres.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'manual_asphalt_rocks')
    args = parser.parse_args()
    if not 0.0 <= args.rocks_edge_width <= 1.0:
        parser.error('--rocks-edge-width must be between 0 and 1 m')
    if not 0 <= args.rocks_flat_border < 1 or (args.rocks_flat_border and not args.rocks_edge_width):
        parser.error('Use 0 <= rocks-flat-border < 1 and a positive rocks-edge-width')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for side in ('right', 'left'):
        right, left = ('t5_large_rocks', 't1_asphalt') if side == 'right' else ('t1_asphalt', 't5_large_rocks')
        grid = (('t1_asphalt', right, right),
                (left, 't2_slippery', right),
                (left, left, 't1_asphalt'))
        generate(args.source, output / f'rocks_{side}.usd', args.overwrite, grid)
        if args.rocks_edge_width:
            taper_rocks(output / f'rocks_{side}.usd', args.rocks_edge_width, args.rocks_flat_border)
            print(f'Rock borders tapered over {args.rocks_edge_width} m: rocks_{side}.usd', flush=True)


if __name__ == '__main__':
    main()
