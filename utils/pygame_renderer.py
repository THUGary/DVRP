from __future__ import annotations
from typing import Tuple, List

try:
    import pygame
except Exception as e:  # pragma: no cover - handled at runtime
    pygame = None  # type: ignore
    _import_error = e
else:
    _import_error = None


Color = Tuple[int, int, int]


class PygameRenderer:
    """Simple Pygame-based renderer for the DVRP grid environment.

    Usage:
      r = PygameRenderer(width, height, cell_size=32)
      r.init()
      while running:
          r.render(obs)  # obs: {time,depot,agent_states,demands,width,height}
      r.close()
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

    def render(self, obs: dict) -> bool:
        """Render one frame. Returns False if the window should close."""
        if not self._inited:
            self.init()
        assert pygame is not None and self._screen is not None and self._font is not None
        if not self._handle_events():
            return False

        self._screen.fill(self.BG)
        # grid origin
        ox = self.margin
        oy = self.margin

        # draw grid cells
        for y in range(self.height):
            for x in range(self.width):
                rx = ox + x * (self.cell_size + self.margin)
                ry = oy + y * (self.cell_size + self.margin)
                pygame.draw.rect(
                    self._screen,
                    self.GRID,
                    pygame.Rect(rx, ry, self.cell_size, self.cell_size),
                    width=1,
                )

        # draw depot
        depot = tuple(obs.get("depot", (0, 0)))
        self._draw_square(depot, color=self.DEPOT)

        # draw demands
        for (dx, dy, dt, dc) in obs.get("demands", []):
            if dt <= obs.get("time", 0):
                self._draw_circle((dx, dy), color=self.DEMAND, radius_ratio=0.35)

        # draw agents
        for idx, (ax, ay, s) in enumerate(obs.get("agent_states", [])):
            self._draw_circle((ax, ay), color=self.AGENT, radius_ratio=0.45)
            # id label
            label = self._font.render(str(idx), True, self.TEXT)
            cx, cy = self._cell_center((ax, ay))
            self._screen.blit(label, (cx - 4, cy - 8))

        # HUD
        hud = self._font.render(
            f"t={obs.get('time', 0)}  demands={len(obs.get('demands', []))}  agents={len(obs.get('agent_states', []))}",
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
