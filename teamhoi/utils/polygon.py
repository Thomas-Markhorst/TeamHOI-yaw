"""Small geometry helper for ordered tabletop grasp outlines."""
import bisect
import math


def sample_rim(vertices, count, z):
    """Uniform arc-length samples around an explicitly inset, CCW outline."""
    if len(vertices) < 3 or count < 3:
        raise ValueError("A rim needs at least three vertices and samples")
    if not math.isfinite(z) or any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in vertices):
        raise ValueError("Rim vertices must be finite [x, y] pairs")
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    area2 = sum(a[0]*b[1] - b[0]*a[1] for a, b in edges)
    if area2 <= 0:
        raise ValueError("Rim vertices must follow a counterclockwise outline")
    ends = [0.0]
    for a, b in edges:
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length <= 0:
            raise ValueError("Rim edges must have positive length")
        ends.append(ends[-1] + length)
    points = []
    for i in range(count):
        distance = i * ends[-1] / count
        edge = min(bisect.bisect_right(ends, distance)-1, len(edges)-1)
        a, b = edges[edge]
        t = (distance-ends[edge]) / (ends[edge+1]-ends[edge])
        points.append([a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), z])
    return points


def asset_ranges(num_envs, num_assets, weights=None):
    """Fixed asset allocation; weighted counts use largest-remainder rounding."""
    if num_assets <= 0 or num_envs < num_assets:
        raise ValueError("Each asset needs at least one environment")
    if weights is None:
        weights = [1.0] * num_assets
    if len(weights) != num_assets or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("asset_weights must contain one positive finite weight per asset")
    exact = [num_envs * w / sum(weights) for w in weights]
    counts = [math.floor(x) for x in exact]
    order = sorted(range(num_assets), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:num_envs - sum(counts)]:
        counts[i] += 1
    if min(counts) == 0:
        raise ValueError("Too few environments for the requested asset_weights")
    result = []
    start = 0
    for count in counts:
        result.append((start, start + count))
        start += count
    return result
