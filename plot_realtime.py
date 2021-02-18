#!/usr/bin/env python3

import argparse
import collections
import json
import socket
import threading
import time
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, port: int, window_s: int):
        self._window_s = window_s
        self._listener = UdpListener(port)
        self._start_time = time.time()
        self._data = collections.deque()
        self._lock = threading.Lock()
        self._update_loop = threading.Thread(daemon=True,
                                             target=self._update_data)
        self._update_loop.start()

        # matplotlib stuff
        self._fig, ((self._acc_ax, self._rot_ax),
                    (self._gyro_ax,
                     self._empty)) = plt.subplots(2,
                                                  2,
                                                  figsize=(19.2, 10.8),
                                                  dpi=100)

        self._rot_lines = []
        self._gyro_lines = []
        self._acc_lines = []

        self._rot_series = 2
        self._ga_series = 3

        fontsize = 24
        legend_fontsize = 16
        x_label = "Time since start (s)"
        for i in range(self._rot_series):
            rot_label = self.ROT[i]
            self._rot_lines.append(
                self._rot_ax.plot([], [], label=rot_label)[0])
            self._rot_ax.set_xlabel(x_label, fontsize=fontsize)
            self._rot_ax.set_ylabel("Rotation (rad)", fontsize=fontsize)
            self._rot_ax.set_xlim([0, self._window_s])
            self._rot_ax.legend(loc="upper left", fontsize=legend_fontsize)

        for i in range(self._ga_series):
            axis = chr(ord("x") + i)
            gyro_label = f"gyro.{axis}"
            acc_label = f"acc.{axis}"

            self._gyro_lines.append(
                self._gyro_ax.plot([], [], label=gyro_label)[0])
            self._gyro_ax.set_xlabel(x_label, fontsize=fontsize)
            self._gyro_ax.set_ylabel("Gyro (rad/s)", fontsize=fontsize)
            self._gyro_ax.set_xlim([0, self._window_s])
            self._gyro_ax.legend(loc="upper left", fontsize=legend_fontsize)

            self._acc_lines.append(
                self._acc_ax.plot([], [], label=acc_label)[0])
            self._acc_ax.set_xlabel(x_label, fontsize=fontsize)
            self._acc_ax.set_ylabel("Acceleration (m/s^2)", fontsize=fontsize)
            self._acc_ax.set_xlim([0, self._window_s])
            self._acc_ax.legend(loc="upper left", fontsize=legend_fontsize)

        self._empty.axis("off")
        self._fig.canvas.set_window_title("MPU-6050")
        self._fig.suptitle("MPU-6050 Time Series", fontsize=fontsize)
        self._fig.tight_layout()

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
        def _update_subplot(ax, timeseries, lines):
            assert len(timeseries) == len(lines)
            for (line, data) in zip(lines, timeseries):
                line.set_xdata(timestamp)
                line.set_ydata(data)

            if timestamp[-1] >= self._window_s:
                ax.set_xlim([timestamp[0], timestamp[-1]])
            else:
                ax.set_xlim([0, self._window_s])

            min_y = np.min(timeseries)
            min_y -= 0.1 * np.abs(min_y)
            max_y = np.max(timeseries)
            max_y += 0.1 * np.abs(max_y)
            ax.set_ylim([min_y, max_y])

        with self._lock:
            data = list(self._data)

        if len(data) <= 1:
            return

        timestamp = np.array([d.timestamp - self._start_time for d in data])

        acc_timeseries = -9.8 * np.array([
            np.array([d.acc.x for d in data]),
            np.array([d.acc.y for d in data]),
            np.array([d.acc.z for d in data]),
        ])
        gyro_timeseries = np.array([
            np.array([d.gyro.x for d in data]),
            np.array([d.gyro.y for d in data]),
            np.array([d.gyro.z for d in data]),
        ])
        rot_timeseries = np.array([
            np.array([d.rot.roll for d in data]),
            np.array([d.rot.pitch for d in data]),
        ])

        _update_subplot(self._rot_ax, rot_timeseries, self._rot_lines)
        _update_subplot(self._gyro_ax, gyro_timeseries, self._gyro_lines)
        _update_subplot(self._acc_ax, acc_timeseries, self._acc_lines)

        self._fig.canvas.draw()

    def close(self):
        self._listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=True,
        help="The UDP port to listen for new data packets on.",
    )

    args = parser.parse_args()

    plt.ion()
    plotter = DataPlotter(args.port, 10)
    while True:
        plotter.update()
