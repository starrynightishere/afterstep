from math import cos, sin


def xor():
    data = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
    queries = [(x, y) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)]
    return data, queries


def regression():
    data = [((x, y), sin(1.5 * x) * cos(0.5 * y) + 0.25 * x * y)
            for x in (-1.0, -0.3, 0.3, 1.0) for y in (-1.0, 0.0, 1.0)]
    queries = [(x, y) for x in (-0.75, 0.0, 0.75) for y in (-0.75, 0.0, 0.75)]
    return data, queries
