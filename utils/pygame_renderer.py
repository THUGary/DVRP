from __future__ import annotations
from typing import Tuple, List, Dict
import math

try:
    import pygame
except Exception as e:  # pragma: no cover - handled at runtime
    pygame = None  # type: ignore
    _import_error = e
else:
    _import_error = None

Color = Tuple[int, int, int]

class PygameRenderer:
    """Pygame 渲染器（对齐新接口）

    render(obs, agent_colors=None, planned_tasks=None)
      - obs: {time,depot,agent_states,demands,width,height}
      - agent_colors: List[RGB] 每个 agent 的颜色
      - planned_tasks: Dict[agent_idx, List[(x,y)]] 将需求节点渲染为与该 agent 相同颜色（圆形）
      - agent: 星形
    """

    def __init__(self, width: int, height: int, cell_size: int = 32, margin: int = 1, caption: str = "DVRP") -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.margin = margin
        self.caption = caption
        self._screen = None
        self._font = None
        self._inited = False

        # colors
        self.BG: Color = (30, 30, 40)
        self.GRID: Color = (60, 60, 70)
        self.DEPOT: Color = (80, 160, 255)
        self.DEMAND: Color = (235, 90, 90)
        self.AGENT: Color = (90, 220, 120)
        self.TEXT: Color = (230, 230, 230)

    def init(self) -> None:
        if self._inited:
            return
        if pygame is None:
            raise RuntimeError(
                f"pygame 导入失败，请先安装 pygame。原始错误: {_import_error}"
            )
        pygame.init()
        w = self.width * self.cell_size + (self.width + 1) * self.margin
        h = self.height * self.cell_size + (self.height + 1) * self.margin + 40  # extra for HUD
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(self.caption)
        self._font = pygame.font.SysFont(None, 18)
        self._inited = True

    def close(self) -> None:
        if pygame and self._inited:
            pygame.quit()
            self._inited = False

    def _handle_events(self) -> bool:
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, obs: dict, agent_colors: List[Color] | None = None, planned_tasks: Dict[int, List[Tuple[int,int]]] | None = None) -> bool:
        """Render one frame. Returns False if the window should close."""
        if not self._inited:
            self.init()
        assert pygame is not None and self._screen is not None and self._font is not None
        if not self._handle_events():
            return False

        self._screen.fill(self.BG)
        # draw grid
        for y in range(self.height):
            for x in range(self.width):
                rx = self.margin + x * (self.cell_size + self.margin)
                ry = self.margin + y * (self.cell_size + self.margin)
                pygame.draw.rect(
                    self._screen,
                    self.GRID,
                    pygame.Rect(rx, ry, self.cell_size, self.cell_size),
                    width=1,
                )

        # draw depot (square)
        depot = tuple(obs.get("depot", (0, 0)))
        self._draw_square(depot, color=self.DEPOT)

        # Build demand color map from planned_tasks
        demand_color: Dict[Tuple[int,int], Color] = {}
        if planned_tasks and agent_colors:
            for i, seq in planned_tasks.items():
                col = agent_colors[i % len(agent_colors)]
                for (tx, ty) in seq:
                    demand_color[(tx, ty)] = col

        # draw demands as circles
        tnow = obs.get("time", 0)
        for d in obs.get("demands", []):
            dx, dy, dt, dc = d[:4]  # (x,y,t,c, ...)
            if dt <= tnow:
                col = demand_color.get((dx, dy), self.DEMAND)
                self._draw_circle((dx, dy), color=col, radius_ratio=0.35)

        # draw agents as stars
        for idx, (ax, ay, s) in enumerate(obs.get("agent_states", [])):
            col = self.AGENT if agent_colors is None else agent_colors[idx % len(agent_colors)]
            self._draw_star((ax, ay), color=col, r_outer=self.cell_size * 0.45, r_inner=self.cell_size * 0.22, n=5)
            # id label
            label = self._font.render(str(idx), True, self.TEXT)
            cx, cy = self._cell_center((ax, ay))
            self._screen.blit(label, (cx - 4, cy - 8))

        # HUD
        hud = self._font.render(
            f"t={tnow}  demands={len(obs.get('demands', []))}  agents={len(obs.get('agent_states', []))}",
            True,
            self.TEXT,
        )
        self._screen.blit(hud, (self.margin, self.height * (self.cell_size + self.margin) + self.margin))

        pygame.display.flip()
        return True

    # --- helpers ---
    def _cell_rect(self, pos: Tuple[int, int]):
        x, y = pos
        rx = self.margin + x * (self.cell_size + self.margin)
        ry = self.margin + y * (self.cell_size + self.margin)
        return pygame.Rect(rx, ry, self.cell_size, self.cell_size)

    def _cell_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r = self._cell_rect(pos)
        return (r.x + r.w // 2, r.y + r.h // 2)

    def _draw_square(self, pos: Tuple[int, int], color: Color) -> None:
        assert pygame is not None and self._screen is not None
        r = self._cell_rect(pos)
        pygame.draw.rect(self._screen, color, r)

    def _draw_circle(self, pos: Tuple[int, int], color: Color, radius_ratio: float = 0.45) -> None:
        assert pygame is not None and self._screen is not None
        cx, cy = self._cell_center(pos)
        radius = int(self.cell_size * radius_ratio)
        pygame.draw.circle(self._screen, color, (cx, cy), radius)

    def _draw_star(self, pos: Tuple[int,int], color: Color, r_outer: float, r_inner: float, n: int = 5) -> None:
        """绘制星形（n=5）"""
        assert pygame is not None and self._screen is not None
        cx, cy = self._cell_center(pos)
        pts: List[Tuple[int,int]] = []
        for i in range(n * 2):
            r = r_outer if i % 2 == 0 else r_inner
            a = math.pi * i / n
            px = int(cx + r * math.sin(a))
            py = int(cy - r * math.cos(a))
            pts.append((px, py))
        pygame.draw.polygon(self._screen, color, pts)