"""NACA 4-digit section generator - example Foilage extension plugin.

Reads the --config JSON written by Foilage, generates the section, and
writes the --output JSON that Foilage turns into the meshed geometry:

    {"ss": [[x, y], ...], "ps": [[x, y], ...]}

SS/PS run LE -> TE; coordinates are in units of the chord (Foilage
normalizes to axial chord = 1 anyway, so any consistent unit works).
Run it standalone for a quick check:

    python generate_naca.py --config config.json --output result.json
"""

import argparse
import json
import math
import sys


def naca4(m, p, t, n=161, clustering=1):
    """SS/PS point lists (LE -> TE, chord = 1) of a NACA MPXX section."""
    def yt(x):                      # thickness distribution (closed TE)
        return 5.0 * t * (0.2969 * math.sqrt(x) - 0.1260 * x
                          - 0.3516 * x * x + 0.2843 * x ** 3
                          - 0.1036 * x ** 4)

    def yc_dy(x):                   # camber line and its slope
        if m <= 0.0 or p <= 0.0:
            return 0.0, 0.0
        if x < p:
            yc = m / (p * p) * (2.0 * p * x - x * x)
            dyc = 2.0 * m / (p * p) * (p - x)
        else:
            yc = m / ((1.0 - p) ** 2) * (1.0 - 2.0 * p + 2.0 * p * x - x * x)
            dyc = 2.0 * m / ((1.0 - p) ** 2) * (p - x)
        return yc, dyc

    # cosine spacing concentrates points at the LE; the exponent pushes
    # the clustering further (1 = plain cosine)
    n = max(int(n), 8)
    beta = [math.pi * i / (n - 1) for i in range(n)]
    xs = [((1.0 - math.cos(b)) / 2.0) ** max(1, int(clustering))
          for b in beta]

    ss, ps = [], []
    for x in xs:
        yc, dyc = yc_dy(x)
        theta = math.atan(dyc)
        th = yt(x)
        ss.append([x - th * math.sin(theta), yc + th * math.cos(theta)])
        ps.append([x + th * math.sin(theta), yc - th * math.cos(theta)])
    return ss, ps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="JSON written by Foilage with the parameters")
    ap.add_argument("--output", required=True,
                    help="where to write the section JSON")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    prm = cfg.get("parameters") or {}
    m = float(prm.get("max_camber", 0.02))
    p = float(prm.get("camber_position", 0.4))
    t = float(prm.get("thickness", 0.12))
    clustering = int(prm.get("n_points_surface", 1))
    n = int(cfg.get("n_points") or 161)

    ss, ps = naca4(m, p, t, n=n, clustering=clustering)
    result = {"ss": ss, "ps": ps,
              "description": f"NACA {round(100*m)}{round(10*p)}{round(100*t):02d}"}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"wrote {args.output} ({len(ss)} points/side)", file=sys.stderr)


if __name__ == "__main__":
    main()
