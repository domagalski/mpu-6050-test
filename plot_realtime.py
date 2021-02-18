#!/usr/bin/env python3

import argparse
import collections
import enum
import json
import socket
import threading
import time
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Rotation(NamedTuple):
    roll: float
    pitch: float


class ThreeVector(NamedTuple):
    """Basic x/y/z vector"""

    x: float
    y: float
    z: float


class Measurement(NamedTuple):
    """MPU 6050 sensor readings"""

    timestamp: float
    temp: float
    rot: Rotation
    gyro: ThreeVector
    acc: ThreeVector

    @classmethod
    def deserialize(cls, data: bytes) -> "Measurement":
        timestamp = time.time()
        json_data = json.loads(data)
        roll = json_data["roll"]
        pitch = json_data["pitch"]
        temp = json_data["temp"]
        rot = Rotation(roll=roll, pitch=pitch)
        gyro = ThreeVector(
            x=json_data["gyro"]["x"],
            y=json_data["gyro"]["y"],
            z=json_data["gyro"]["z"],
        )
        acc = ThreeVector(
            x=json_data["acc"]["x"],
            y=json_data["acc"]["y"],
            z=json_data["acc"]["z"],
        )
        return Measurement(
            timestamp=timestamp,
            temp=temp,
            rot=rot,
            gyro=gyro,
            acc=acc,
        )


class MeasType(enum.Enum):
    temp = 0
    rot = 1
    gyro = 2
    acc = 3


class UdpListener:
    """Listen for data over UDP"""
    def __init__(self, port: int):
        self._port = port
        self._addr = ("0.0.0.0", port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1)
        self._sock.bind(self._addr)
        print(f"Listening on UDP port: {port}")

    def get(self) -> Measurement:
        """Listen for data and parse it"""
        raw_data = self._sock.recvfrom(1024)
        return Measurement.deserialize(raw_data[0])

    def close(self):
        self._sock.close()


class DataPlotter:
    """Animate incoming data"""

    ROT = ["roll", "pitch"]

    def __init__(self, meas_type: MeasType, port: int, window_s: int):
        self._meas_type = meas_type
        self._window_s = window_s
        self._listener = UdpListener(port)
        self._start_time = time.time()
        self._data = collections.deque()
        self._lock = threading.Lock()
        self._update_loop = threading.Thread(daemon=True,
                                             target=self._update_data)
        self._update_loop.start()

        # matplotlib stuff
        self._fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        self._ax = plt.axes()
        self._lines = []

        if self._meas_type in [MeasType.acc, MeasType.gyro]:
            n_series = 3
        elif self._meas_type == MeasType.rot:
            n_series = 2
        else:
            n_series = 1

        for i in range(n_series):
            if self._meas_type in [MeasType.acc, MeasType.gyro]:
                axis = chr(ord("x") + i)
                label = f"{self._meas_type.name}.{axis}"
            elif self._meas_type == MeasType.rot:
                label = self.ROT[i]
            else:
                label = self._meas_type.name

            if self._meas_type == MeasType.temp:
                y_label = "Temperature (C)"
            elif self._meas_type == MeasType.rot:
                y_label = "Rotation (rad)"
            elif self._meas_type == MeasType.gyro:
                y_label = "Gyro (rad/s)"
            elif self._meas_type == MeasType.acc:
                y_label = "Acceleration (m/s^2)"
            else:
                raise NotImplementedError

            self._lines.append(self._ax.plot([], [], label=label)[0])

        self._ax.legend(loc="upper left", fontsize=20)
        fontsize = 32
        self._ax.set_xlabel("Time since start (s)", fontsize=fontsize)
        self._ax.set_ylabel(y_label, fontsize=fontsize)
        self._ax.set_title("MPU-6050 Time Series", fontsize=fontsize)
        self._ax.set_xlim([0, self._window_s])

    def _update_data(self):
        while True:
            try:
                point = self._listener.get()
            except socket.timeout:
                continue
            with self._lock:
                self._update_data_while_locked(point)

    def _update_data_while_locked(self, point: Measurement):
        self._data.append(point)
        now = time.time()
        while len(self._data):
            if self._data[0].timestamp < now - self._window_s:
                self._data.popleft()
            else:
                break

    def update(self):
        with self._lock:
            data = list(self._data)

        if len(data) <= 1:
            return

        timestamp = np.array([d.timestamp - self._start_time for d in data])
        if self._meas_type in [MeasType.acc, MeasType.gyro]:
            timeseries = np.array([
                np.array([
                    d.__getattribute__(self._meas_type.name).x for d in data
                ]),
                np.array([
                    d.__getattribute__(self._meas_type.name).y for d in data
                ]),
                np.array([
                    d.__getattribute__(self._meas_type.name).z for d in data
                ]),
            ])
        elif self._meas_type == MeasType.rot:
            timeseries = []
            for name in self.ROT:
                timeseries.append(
                    np.array([
                        d.__getattribute__(
                            self._meas_type.name).__getattribute__(name)
                        for d in data
                    ]))
            timeseries = np.array(timeseries)
        else:
            timeseries = np.array([
                np.array(
                    [d.__getattribute__(self._meas_type.name) for d in data]),
            ])

        if self._meas_type == MeasType.acc:
            timeseries *= -9.8

        assert len(timeseries) == len(self._lines)
        for (line, data) in zip(self._lines, timeseries):
            line.set_xdata(timestamp)
            line.set_ydata(data)

        if timestamp[-1] >= self._window_s:
            self._ax.set_xlim([timestamp[0], timestamp[-1]])
        else:
            self._ax.set_xlim([0, self._window_s])

        min_y = np.min(timeseries)
        min_y -= 0.1 * np.abs(min_y)
        max_y = np.max(timeseries)
        max_y += 0.1 * np.abs(max_y)
        self._ax.set_ylim([min_y, max_y])

        self._fig.canvas.draw()

    def close(self):
        self._listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--measurement",
        choices=[m.name for m in list(MeasType)],
        required=True,
        help="The measurement type to view.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=True,
        help="The UDP port to listen for new data packets on.",
    )

    args = parser.parse_args()

    plt.ion()
    plotter = DataPlotter(MeasType[args.measurement], args.port, 10)
    while True:
        plotter.update()
