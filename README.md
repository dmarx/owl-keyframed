# TAKEAWAYS

1. The added overhead of waiting for the workflow to do its thing is non-trivial. Not super conducive to real-time collaboration. 
  - This could be a good approach for automating suggested improvements, but the "pair-programming" workflow of copilot probably makes more sense for more real-time type stuff.
2. File extension isn't a good trigger. Just in this simple experiment, I had three different filetypes I wanted to generate (.py, .md, .yaml), and this was a fairly trivial example.
  - would be better if something like this could be triggered on its own without me needing to prompt it.
    - automated code review suggestions
    - automated docstring completions
    - automated type hinting
    - automated readme/tutorial/documentation generation
      - would require decomposition into several steps
3. Context length is a significant limiting factor. 
  - could probably prompt it to generate smaller chunks of self-contained code (e.g. one class per file)
  - a language like c++ that uses a separate header file to define interfaces might be well suited for this.
  - for python, maybe I could get away with generating typing stubs for context.

- dmarx

---

# Keyframed

Keyframed is a python library that provides a set of datatypes for specifying and manipulating curves parameterized by keyframes and interpolators. It is designed to be easy to use and integrate into existing projects.

## Installation

To install Keyframed, simply use pip:

```
pip install keyframed
```

## Usage

To use Keyframed, import the relevant datatypes from the `keyframed` module:

```python
from keyframed import Keyframe, Interpolator, Curve
```

### Keyframes

A `Keyframe` represents a point on a curve at a specific time. It consists of a time value and a value for the curve at that time:

```python
kf = Keyframe(time=0.0, value=1.0)
```

### Interpolators

An `Interpolator` defines how the curve should be interpolated between keyframes. Keyframed provides several built-in interpolators, including linear, cubic, and bezier:

```python
from keyframed.interpolators import LinearInterpolator, CubicInterpolator, BezierInterpolator

linear = LinearInterpolator()
cubic = CubicInterpolator()
bezier = BezierInterpolator(control_points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
```

### Curves

A `Curve` is a collection of keyframes and an interpolator. It represents a complete curve:

```python
curve = Curve(keyframes=[Keyframe(time=0.0, value=1.0), Keyframe(time=1.0, value=0.0)], interpolator=linear)
```

### Evaluating Curves

To evaluate a curve at a specific time, use the `evaluate` method:

```python
value = curve.evaluate(time=0.5)
```

### Modifying Curves

Curves can be modified by adding or removing keyframes:

```python
curve.add_keyframe(Keyframe(time=0.5, value=0.5))
curve.remove_keyframe(index=1)
```

### Serialization

Curves can be serialized to and deserialized from JSON:

```python
import json

json_data = json.dumps(curve.to_dict())
curve = Curve.from_dict(json.loads(json_data))
```

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

Keyframed is licensed under the [MIT License](LICENSE).
