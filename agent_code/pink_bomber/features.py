import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BOMB_TIMER = 4   # steps until a dropped bomb explodes (see settings.py)
BOMB_POWER = 3   # blast radius in each direction (see settings.py)


def get_danger_map(field, bombs, explosion_map):
    danger = set()
    width, height = field.shape

    xs, ys = np.where(explosion_map > 0)
    for x, y in zip(xs, ys):
        danger.add((x, y))

    for (bx, by), timer in bombs:
        danger.add((bx, by))
        for dx, dy in DIRECTIONS:
            for step in range(1, BOMB_POWER + 1):
                nx, ny = bx + dx * step, by + dy * step
                if not (0 <= nx < width and 0 <= ny < height):
                    break
                if field[nx, ny] == -1:
                    break
                danger.add((nx, ny))
    return danger


def bfs_escape_exists(field, danger, occupied, start, max_steps):
    """Is there a walkable path of length <= max_steps from start to a tile outside danger?"""
    width, height = field.shape
    visited = {start}
    frontier = deque([(start, 0)])
    while frontier:
        (x, y), d = frontier.popleft()
        if d >= max_steps:
            continue
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if field[nx, ny] != 0:  # wall or crate blocks movement
                continue
            if (nx, ny) in occupied:  # a bomb sitting there also blocks movement
                continue
            visited.add((nx, ny))
            if (nx, ny) not in danger:
                return True
            frontier.append(((nx, ny), d + 1))
    return False


def is_safe_to_bomb(field, bombs, x, y):
    """Would dropping a bomb at (x,y) still leave an escape route?"""
    simulated_bombs = list(bombs) + [((x, y), BOMB_TIMER - 1)]
    danger = get_danger_map(field, simulated_bombs, np.zeros_like(field))
    occupied = {b[0] for b in simulated_bombs}
    return 1.0 if bfs_escape_exists(field, danger, occupied, (x, y), BOMB_TIMER) else 0.0


def nearest_direction(x, y, targets):
    direction = [0.0, 0.0, 0.0, 0.0]
    if not targets:
        return direction
    nearest = min(targets, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    dx, dy = nearest[0] - x, nearest[1] - y
    if dx == 0 and dy == 0:
        return direction
    if abs(dx) > abs(dy):
        direction[1 if dx > 0 else 3] = 1.0
    else:
        direction[2 if dy > 0 else 0] = 1.0
    return direction


def adjacent_to_crate(field, x, y):
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 1:
            return 1.0
    return 0.0


def state_to_features(game_state: dict) -> np.ndarray:
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    _, _, bomb_available, (x, y) = game_state['self']

    width, height = field.shape
    danger = get_danger_map(field, bombs, explosion_map)

    features = []

    # 1-4: can I move in each direction?
    occupied = {b[0] for b in bombs}
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        walkable = (
            0 <= nx < width and 0 <= ny < height
            and field[nx, ny] == 0
            and (nx, ny) not in occupied
        )
        features.append(1.0 if walkable else 0.0)

    # 5: can I drop a bomb right now?
    features.append(1.0 if bomb_available else 0.0)

    # 6-9: is the tile in each direction dangerous?
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        features.append(1.0 if (nx, ny) in danger else 0.0)

    # 10: am I standing in danger right now?
    features.append(1.0 if (x, y) in danger else 0.0)

    # 11-14: direction toward nearest coin
    features.extend(nearest_direction(x, y, coins))

    # 15-18: direction toward nearest crate
    xs, ys = np.where(field == 1)
    crate_positions = list(zip(xs.tolist(), ys.tolist()))
    features.extend(nearest_direction(x, y, crate_positions))

    # 19: is there a crate right next to me?
    features.append(adjacent_to_crate(field, x, y))

    # 20: would dropping a bomb here still leave me an escape route?
    features.append(is_safe_to_bomb(field, bombs, x, y) if bomb_available else 0.0)

    return np.array(features, dtype=np.float32)
