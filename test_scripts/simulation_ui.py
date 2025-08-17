import tkinter as tk
import rospy
from turtlesim.srv import TeleportAbsolute
import numpy as np
import math


class Clicker:
    def __init__(
        self, pl: list = [], offsets: tuple = (), convertion_scale: float = 10.0
    ):
        self.circ = None
        self.srv_px = [
            rospy.ServiceProxy(f"/change_element/F{a}", TeleportAbsolute)
            for a in range(0, 3)
        ]
        self.pl = pl
        self.offset = offset
        self.convertion_scale = convertion_scale

    def draw_stuff():
        """TODO: Draw canvas here"""
        pass

    @staticmethod
    def calc_diff(a, b):
        summer = sum([(ae - be) ** 2 for ae, be in zip(a, b)])
        return math.sqrt(summer)

    @staticmethod
    def calc_Fs(Fz, radl, el):
        return Fz * (1 / (radl[el] / radl[0] + radl[el] / radl[1] + radl[el] / radl[2]))

    def left_click(self, event):
        if self.circ is not None:
            canvas.delete(self.circ)
        self.circ = canvas.create_oval(
            event.x + 2,
            event.y + 2,
            event.x - 2,
            event.y - 2,
            fill="green",
        )
        T.config(
            text=f"({(event.x - self.offset[0])/10}, {(event.y - self.offset[1])/10})"
        )
        z_val = w.get()
        diffs = [
            self.calc_diff((event.x - self.offset[0], event.y - self.offset[1]), p)
            for p in self.pl
        ]
        print(diffs)

        for i, prox in enumerate(self.srv_px):
            F = self.calc_Fs(z_val, diffs, i)
            prox.call(
                0,
                0,
                F,
            )


if __name__ == "__main__":
    rospy.init_node("click_canvas")
    root = tk.Tk()

    canvas = tk.Canvas(root, width="400", height="300")

    OVALSIZE = 20
    offset = (25, 25)

    p1 = (0, 0)
    p2 = (250, 0)
    p3 = (250, 250)
    p4 = (0, 250)
    pl = [p1, p2, p3, p4]
    cl = Clicker(pl=pl, offsets=offset)

    for p in pl:
        canvas.create_oval(
            offset[0] + p[0],
            offset[1] + p[1],
            offset[0] + p[0] + OVALSIZE,
            offset[1] + p[1] + OVALSIZE,
            fill="red",
        )
    canvas.pack()

    w = tk.Scale(root, from_=1, to=60, orient=tk.HORIZONTAL)
    T = tk.Label(root, height=2, width=30, text=f"({0},{0})")
    T.pack()
    w.pack()

    canvas.bind("<Button-1>", cl.left_click)

    root.mainloop()
