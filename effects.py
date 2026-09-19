"""Procedural visual effects for the gesture tracker."""
import math
import random
from dataclasses import dataclass

import cv2
import numpy as np


ORANGE = (25, 145, 255)
GOLD = (70, 225, 255)
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    size: int

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 75 * dt
        self.vx *= 0.985
        self.vy *= 0.985
        self.life -= dt


class SpellEffects:
    """Maintains air-drawn strokes, sparks, and animated energy rings."""

    def __init__(self):
        self.strokes = []
        self.active_stroke = []
        self.particles = []
        self.previous_gesture = ""
        self.ring_charge = 0.0
        self.angle = 0.0

    def clear(self):
        self.strokes.clear()
        self.active_stroke.clear()
        self.particles.clear()

    def finish_stroke(self):
        if len(self.active_stroke) > 1:
            self.strokes.append(self.active_stroke[:])
            self.strokes = self.strokes[-24:]
        self.active_stroke.clear()

    def emit_sparks(self, origin, count=5, power=135):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(power * 0.35, power)
            self.particles.append(Particle(
                origin[0], origin[1], math.cos(angle) * speed,
                math.sin(angle) * speed, random.uniform(0.25, 0.75),
                random.randint(1, 4)))
        self.particles = self.particles[-450:]

    def update(self, gesture, index_tip, palm_center, dt):
        self.angle = (self.angle + 95 * dt) % 360

        # A pinch acts like touching a pen to the canvas. Releasing the pinch
        # lifts the pen and stores the completed stroke.
        if gesture == "PINCH":
            if not self.active_stroke or math.dist(self.active_stroke[-1], index_tip) >= 4:
                self.active_stroke.append(index_tip)
                self.active_stroke = self.active_stroke[-900:]
                self.emit_sparks(index_tip, 2, 70)
        else:
            self.finish_stroke()

        if gesture == "PINCH":
            if self.previous_gesture != "PINCH":
                self.emit_sparks(index_tip, 42, 320)

        target_charge = 1.0 if gesture == "OPEN PALM" else 0.0
        speed = 2.3 if target_charge else 3.5
        self.ring_charge += (target_charge - self.ring_charge) * min(1.0, speed * dt)
        if gesture == "OPEN PALM" and random.random() < 0.55:
            radius = 45 + 70 * self.ring_charge
            a = random.uniform(0, math.tau)
            point = (int(palm_center[0] + math.cos(a) * radius),
                     int(palm_center[1] + math.sin(a) * radius))
            self.emit_sparks(point, 1, 45)

        for particle in self.particles:
            particle.update(dt)
        self.particles = [particle for particle in self.particles if particle.life > 0]
        self.previous_gesture = gesture

    @staticmethod
    def _polyline(layer, points, color, thickness):
        if len(points) > 1:
            cv2.polylines(layer, [np.asarray(points, dtype=np.int32)], False,
                          color, thickness, cv2.LINE_AA)

    def _draw_rune(self, layer, center):
        charge = self.ring_charge
        if charge < 0.04:
            return
        radius = int((45 + 70 * charge))
        thickness = max(1, int(3 * charge))
        cv2.circle(layer, center, radius, GOLD, thickness, cv2.LINE_AA)
        cv2.circle(layer, center, int(radius * 0.72), ORANGE, thickness, cv2.LINE_AA)

        for offset in range(0, 360, 45):
            a = math.radians(offset + self.angle)
            b = math.radians(offset + 22 + self.angle)
            outer = (int(center[0] + math.cos(a) * radius),
                     int(center[1] + math.sin(a) * radius))
            inner = (int(center[0] + math.cos(b) * radius * 0.72),
                     int(center[1] + math.sin(b) * radius * 0.72))
            cv2.line(layer, outer, inner, GOLD, thickness, cv2.LINE_AA)

        triangle = []
        for offset in (270, 30, 150):
            a = math.radians(offset - self.angle * 0.55)
            triangle.append((int(center[0] + math.cos(a) * radius * 0.52),
                             int(center[1] + math.sin(a) * radius * 0.52)))
        cv2.polylines(layer, [np.asarray(triangle, dtype=np.int32)], True,
                      ORANGE, thickness, cv2.LINE_AA)

    def render(self, frame, palm_center=None, art=None, composite=True):
        if art is None:
            art = np.zeros_like(frame)
        for stroke in self.strokes:
            self._polyline(art, stroke, ORANGE, 4)
        self._polyline(art, self.active_stroke, GOLD, 5)

        if palm_center is not None:
            self._draw_rune(art, palm_center)

        for particle in self.particles:
            fade = max(0.0, min(1.0, particle.life / 0.75))
            color = tuple(int(channel * fade) for channel in GOLD)
            start = (int(particle.x), int(particle.y))
            end = (int(particle.x - particle.vx * 0.025),
                   int(particle.y - particle.vy * 0.025))
            cv2.line(art, start, end, color, particle.size, cv2.LINE_AA)

        if not composite:
            return art
        self.composite(frame, art)

    @staticmethod
    def composite(frame, art):
        # Blur at quarter resolution; preserve crisp strokes at full resolution.
        h, w = frame.shape[:2]
        small = cv2.resize(art, (max(1,w//4), max(1,h//4)))
        glow_large = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.5), (w,h))
        glow_small = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1), (w,h))
        cv2.addWeighted(frame, 1.0, glow_large, 0.65, 0, frame)
        cv2.addWeighted(frame, 1.0, glow_small, 0.85, 0, frame)
        cv2.addWeighted(frame, 1.0, art, 1.0, 0, frame)
