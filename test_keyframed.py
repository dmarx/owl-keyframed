import pytest
from keyframed import Keyframe, Interpolator, LinearInterpolator, CubicInterpolator, BezierInterpolator, Curve

def test_keyframe():
    kf = Keyframe(1.0, 2.0)
    assert kf.time == 1.0
    assert kf.value == 2.0

def test_linear_interpolator():
    li = LinearInterpolator()
    assert li.interpolate(0.0, (0.0, 0.0), (1.0, 1.0)) == 0.0
    assert li.interpolate(0.5, (0.0, 0.0), (1.0, 1.0)) == 0.5
    assert li.interpolate(1.0, (0.0, 0.0), (1.0, 1.0)) == 1.0

def test_cubic_interpolator():
    ci = CubicInterpolator()
    assert ci.interpolate(0.0, (0.0, 0.0), (1.0, 1.0)) == 0.0
    assert ci.interpolate(0.5, (0.0, 0.0), (1.0, 1.0)) == 0.5
    assert ci.interpolate(1.0, (0.0, 0.0), (1.0, 1.0)) == 1.0

def test_bezier_interpolator():
    bi = BezierInterpolator([(0.25, 0.25), (0.75, 0.75)])
    assert bi.interpolate(0.0, (0.0, 0.0), (1.0, 1.0)) == 0.0
    assert bi.interpolate(0.5, (0.0, 0.0), (1.0, 1.0)) == 0.5
    assert bi.interpolate(1.0, (0.0, 0.0), (1.0, 1.0)) == 1.0

def test_curve():
    li = LinearInterpolator()
    ci = CubicInterpolator()
    bi = BezierInterpolator([(0.25, 0.25), (0.75, 0.75)])
    kf1 = Keyframe(0.0, 0.0)
    kf2 = Keyframe(1.0, 1.0)
    kf3 = Keyframe(2.0, 0.0)
    curve = Curve([kf1, kf2, kf3], li)
    assert curve.evaluate(-1.0) == 0.0
    assert curve.evaluate(0.0) == 0.0
    assert curve.evaluate(0.5) == 0.5
    assert curve.evaluate(1.0) == 1.0
    assert curve.evaluate(2.0) == 0.0
    curve.add_keyframe(Keyframe(1.5, 0.5))
    assert curve.evaluate(1.25) == 0.25
    curve.remove_keyframe(1)
    assert curve.evaluate(1.25) == 0.5
    data = curve.to_dict()
    assert data == {"keyframes": [{"time": 0.0, "value": 0.0}, {"time": 1.0, "value": 1.0}, {"time": 1.5, "value": 0.5}, {"time": 2.0, "value": 0.0}], "interpolator": {"type": "LinearInterpolator", "params": {}}}
    curve2 = Curve.from_dict(data)
    assert curve2.evaluate(0.5) == 0.5
    assert curve2.evaluate(1.25) == 0.25
    assert curve2.evaluate(2.0) == 0.0


# The following code provides demonstrative usage examples for the `Keyframe`, `Interpolator`, `LinearInterpolator`, `CubicInterpolator`, `BezierInterpolator`, and `Curve` classes:

# from keyframed import Keyframe, Interpolator, LinearInterpolator, CubicInterpolator, BezierInterpolator, Curve

# # Create a keyframe
# kf = Keyframe(1.0, 2.0)

# # Create an interpolator
# li = LinearInterpolator()

# # Interpolate between two points
# value = li.interpolate(0.5, (0.0, 0.0), (1.0, 1.0))

# # Create a curve
# kf1 = Keyframe(0.0, 0.0)
# kf2 = Keyframe(1.0, 1.0)
# kf3 = Keyframe(2.0, 0.0)
# curve = Curve([kf1, kf2, kf3], li)

# # Evaluate the curve at a specific time
# value = curve.evaluate(0.5)

# Add a keyframe to the curve
curve.add_keyframe(Keyframe(1.5, 0.5))

# Remove a keyframe from the curve
curve.remove_keyframe(1)

# Serialize the curve to a dictionary
data = curve.to_dict()

# Deserialize a curve from a dictionary
curve2 = Curve.from_dict(data)
