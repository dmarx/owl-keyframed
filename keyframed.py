import json
from typing import List, Tuple

class Keyframe:
    """
    A `Keyframe` represents a point on a curve at a specific time. It consists of a time value and a value for the curve at that time.
    """
    def __init__(self, time: float, value: float):
        self.time = time
        self.value = value

class Interpolator:
    """
    An `Interpolator` defines how the curve should be interpolated between keyframes.
    """
    def __init__(self):
        pass

    def interpolate(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        """
        Interpolate between two points.

        Args:
            t (float): The interpolation factor.
            p0 (Tuple[float, float]): The first point.
            p1 (Tuple[float, float]): The second point.

        Returns:
            float: The interpolated value.
        """
        pass

class LinearInterpolator(Interpolator):
    """
    A linear interpolator.
    """
    def interpolate(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        return p0[1] * (1.0 - t) + p1[1] * t

class CubicInterpolator(Interpolator):
    """
    A cubic interpolator.
    """
    def interpolate(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        dt = p1[0] - p0[0]
        dt2 = dt * dt
        m0 = (p1[1] - p0[1]) / dt
        m1 = 0.0
        if len(p0) == 3 and len(p1) == 3:
            m1 = ((p1[2] - p0[2]) / dt2 - m0) * dt
        t2 = t * t
        t3 = t2 * t
        return (2.0 * t3 - 3.0 * t2 + 1.0) * p0[1] + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * p1[1] + (t3 - t2) * m1

class BezierInterpolator(Interpolator):
    """
    A bezier interpolator.
    """
    def __init__(self, control_points: List[Tuple[float, float]]):
        super().__init__()
        self.control_points = control_points

    def interpolate(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        def bezier(t: float, p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
            return (1.0 - t) ** 3 * p0[1] + 3.0 * (1.0 - t) ** 2 * t * p1[1] + 3.0 * (1.0 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]

        t0 = 0.0
        t1 = 1.0
        while True:
            p01 = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            p12 = (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)
            p23 = (p2[0] + (p3[0] - p2[0]) * t, p2[1] + (p3[1] - p2[1]) * t)
            p012 = (p01[0] + (p12[0] - p01[0]) * t, p01[1] + (p12[1] - p01[1]) * t)
            p123 = (p12[0] + (p23[0] - p12[0]) * t, p12[1] + (p23[1] - p12[1]) * t)
            p0123 = (p012[0] + (p123[0] - p012[0]) * t, p012[1] + (p123[1] - p012[1]) * t)
            if abs(p0123[0] - p0[0]) < 1e-6:
                break
            if p0123[0] > p0[0]:
                t1 = t
            else:
                t0 = t
            t = (t0 + t1) / 2.0
        return bezier(t, p0, self.control_points[0], self.control_points[1], p1)

class Curve:
    """
    A `Curve` is a collection of keyframes and an interpolator. It represents a complete curve.
    """
    def __init__(self, keyframes: List[Keyframe], interpolator: Interpolator):
        self.keyframes = keyframes
        self.interpolator = interpolator

    def evaluate(self, time: float) -> float:
        """
        Evaluate the curve at a specific time.

        Args:
            time (float): The time to evaluate the curve at.

        Returns:
            float: The value of the curve at the given time.
        """
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        for i in range(len(self.keyframes) - 1):
            if time >= self.keyframes[i].time and time <= self.keyframes[i + 1].time:
                t = (time - self.keyframes[i].time) / (self.keyframes[i + 1].time - self.keyframes[i].time)
                return self.interpolator.interpolate(t, (self.keyframes[i].time, self.keyframes[i].value), (self.keyframes[i + 1].time, self.keyframes[i + 1].value))

    def add_keyframe(self, keyframe: Keyframe):
        """
        Add a keyframe to the curve.

        Args:
            keyframe (Keyframe): The keyframe to add.
        """
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda kf: kf.time)

    def remove_keyframe(self, index: int):
        """
        Remove a keyframe from the curve.

        Args:
            index (int): The index of the keyframe to remove.
        """
        del self.keyframes[index]

    def to_dict(self) -> dict:
        """
        Serialize the curve to a dictionary.

        Returns:
            dict: The serialized curve.
        """
        return {
            "keyframes": [{"time": kf.time, "value": kf.value} for kf in self.keyframes],
            "interpolator": {"type": type(self.interpolator).__name__, "params": self.interpolator.__dict__},
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Curve':
        """
        Deserialize a curve from a dictionary.

        Args:
            data (dict): The serialized curve.

        Returns:
            Curve: The deserialized curve.
        """
        interpolator_type = globals()[data["interpolator"]["type"]]
        interpolator_params = data["interpolator"]["params"]
        interpolator = interpolator_type(**interpolator_params)
        keyframes = [Keyframe(kf["time"], kf["value"]) for kf in data["keyframes"]]
        return cls(keyframes, interpolator)
