#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""贪吃蛇 (Snake) —— 纯 Python 标准库 tkinter 实现，无第三方依赖。

操作说明:
    ↑ / ↓ / ← / →   或   W / A / S / D    控制方向
    空格              暂停 / 继续
    T                 切换穿墙模式（开：穿墙而过；关：撞墙结束）
    R                 重新开始
    Esc / Q           退出

最高分保存在脚本同目录的 highscore.json 中，自动读写。

运行:
    python snake.py
"""

import json
import os
import random
import tkinter as tk

# ---------- 可调参数 ----------
CELL = 20          # 每格边长（像素）
COLS = 30          # 横向格数
ROWS = 20          # 纵向格数
START_SPEED = 120  # 初始移动间隔（毫秒），越小越快
MIN_SPEED = 60     # 最快速度
SPEED_STEP = 4     # 每吃一个食物加快的毫秒数
FOOD_SCORE = 10    # 吃一个食物的得分
WRAP_DEFAULT = False  # 初始是否开启穿墙模式

# 最高分存档：脚本所在目录下的 highscore.json
SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.json")

# ---------- 配色 ----------
BG = "#1e1e1e"
GRID = "#2a2a2a"
SNAKE = "#4caf50"
SNAKE_HEAD = "#8bc34a"
FOOD = "#e53935"
TEXT = "#eeeeee"
WRAP_BORDER = "#ffb300"  # 穿墙模式下的边框颜色


def load_high_score() -> int:
    """读取历史最高分；文件不存在或损坏时返回 0。"""
    try:
        with open(SCORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("high_score", 0))
    except (OSError, ValueError, TypeError):
        return 0


def save_high_score(score: int) -> None:
    """把最高分写入存档文件；失败时静默忽略（不影响游戏）。"""
    try:
        with open(SCORE_FILE, "w", encoding="utf-8") as f:
            json.dump({"high_score": int(score)}, f, ensure_ascii=False)
    except OSError:
        pass


class SnakeGame:
    """一个贪吃蛇游戏实例，负责状态、渲染与主循环。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("贪吃蛇 - Snake")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=COLS * CELL,
            height=ROWS * CELL,
            bg=BG,
            highlightthickness=0,
        )
        self.canvas.pack()

        self.score_var = tk.StringVar()
        self.label = tk.Label(
            root, textvariable=self.score_var,
            font=("Consolas", 12), bg=BG, fg=TEXT,
        )
        self.label.pack(fill="x")

        self.root.bind("<Key>", self.on_key)

        self.high_score = load_high_score()  # 历史最高分
        self.wrap = WRAP_DEFAULT             # 穿墙模式开关（跨重开保留）
        self._job = None  # after() 定时器句柄
        self.reset()

    # ---------- 状态管理 ----------
    def reset(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None

        cx, cy = COLS // 2, ROWS // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = (1, 0)   # 当前方向 (dx, dy)
        self.pending = (1, 0)     # 下一帧待应用的方向
        self.food = self.spawn_food()
        self.score = 0
        self.speed = START_SPEED
        self.paused = False
        self.over = False

        self.update_label()
        self.draw()
        self.schedule()

    def schedule(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(self.speed, self.tick)

    # ---------- 输入 ----------
    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        mapping = {
            "up": (0, -1), "w": (0, -1),
            "down": (0, 1), "s": (0, 1),
            "left": (-1, 0), "a": (-1, 0),
            "right": (1, 0), "d": (1, 0),
        }
        if key in mapping:
            new_dir = mapping[key]
            # 禁止 180° 掉头（相对当前方向）
            if new_dir != (-self.direction[0], -self.direction[1]):
                self.pending = new_dir
        elif key == "space":
            if not self.over:
                self.paused = not self.paused
                self.draw()
        elif key == "t":
            # 穿墙模式可随时切换（即使游戏结束也能切，方便下一局）
            self.wrap = not self.wrap
            self.update_label()
            self.draw()
        elif key == "r":
            self.reset()
        elif key in ("escape", "q"):
            self.root.destroy()

    # ---------- 主循环 ----------
    def tick(self) -> None:
        self._job = None
        if not self.over and not self.paused:
            self.direction = self.pending
            self.step()
            self.draw()
        if not self.over:
            self.schedule()

    def step(self) -> None:
        hx, hy = self.snake[0]
        dx, dy = self.direction
        nx, ny = hx + dx, hy + dy

        # 越界处理：穿墙模式则从对面出现，否则撞墙结束
        if not (0 <= nx < COLS and 0 <= ny < ROWS):
            if self.wrap:
                nx %= COLS
                ny %= ROWS
            else:
                self.game_over()
                return

        # 撞到自己（若这步会吃到食物则尾巴不动，需把尾巴也算进去）
        body = self.snake if (nx, ny) == self.food else self.snake[:-1]
        if (nx, ny) in body:
            self.game_over()
            return

        self.snake.insert(0, (nx, ny))
        if (nx, ny) == self.food:
            self.score += FOOD_SCORE
            self.speed = max(MIN_SPEED, self.speed - SPEED_STEP)
            self.food = self.spawn_food()
            self.update_label()
        else:
            self.snake.pop()

    def spawn_food(self):
        """在所有空格中随机选一个放食物；棋盘填满时返回 None。"""
        free = [
            (x, y)
            for x in range(COLS)
            for y in range(ROWS)
            if (x, y) not in self.snake
        ]
        return random.choice(free) if free else None

    def game_over(self) -> None:
        self.over = True

        # 刷新最高分并落盘
        new_record = self.score > self.high_score
        if new_record:
            self.high_score = self.score
            save_high_score(self.high_score)

        self.draw()
        center_x, center_y = COLS * CELL // 2, ROWS * CELL // 2
        tip = "🎉 新纪录！" if new_record else f"最高分: {self.high_score}"
        self.canvas.create_text(
            center_x, center_y,
            text=(f"游戏结束\n得分: {self.score}\n{tip}\n按 R 重新开始"),
            fill=TEXT, font=("Consolas", 20, "bold"), justify="center",
        )

    # ---------- 渲染 ----------
    def draw(self) -> None:
        self.canvas.delete("all")

        # 网格
        for x in range(0, COLS * CELL + 1, CELL):
            self.canvas.create_line(x, 0, x, ROWS * CELL, fill=GRID)
        for y in range(0, ROWS * CELL + 1, CELL):
            self.canvas.create_line(0, y, COLS * CELL, y, fill=GRID)

        # 穿墙模式下高亮边框，提示"可以穿过去"
        if self.wrap:
            self.canvas.create_rectangle(
                2, 2, COLS * CELL - 2, ROWS * CELL - 2,
                outline=WRAP_BORDER, width=3,
            )

        # 食物
        if self.food is not None:
            self.draw_cell(self.food, FOOD)

        # 蛇（头颜色不同）
        for i, cell in enumerate(self.snake):
            self.draw_cell(cell, SNAKE_HEAD if i == 0 else SNAKE)

        # 暂停提示
        if self.paused and not self.over:
            self.canvas.create_text(
                COLS * CELL // 2, ROWS * CELL // 2,
                text="已暂停\n按空格继续",
                fill=TEXT, font=("Consolas", 20, "bold"), justify="center",
            )

    def draw_cell(self, cell, color: str) -> None:
        x, y = cell
        self.canvas.create_rectangle(
            x * CELL + 1, y * CELL + 1,
            (x + 1) * CELL - 1, (y + 1) * CELL - 1,
            fill=color, outline="",
        )

    def update_label(self) -> None:
        mode = "穿墙:开" if self.wrap else "穿墙:关"
        self.score_var.set(
            f"得分: {self.score}    最高分: {self.high_score}    "
            f"长度: {len(self.snake)}    [{mode}]    "
            f"方向键/WASD 移动 · 空格暂停 · T 穿墙 · R 重开"
        )


def main() -> None:
    root = tk.Tk()
    SnakeGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
