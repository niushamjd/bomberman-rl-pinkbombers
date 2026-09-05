import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Relative offsets for UP, RIGHT, DOWN, LEFT
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def get_danger_map(field, bombs, explosion_map):
    """
    Returns a set of (x, y) coordinates that are currently dangerous:
    either an active explosion, or a tile that will be hit once a
    ticking bomb explodes.
    """
    danger = set()
    width, height = field.shape

    # Current explosions
    xs, ys = np.where(explosion_map > 0)
    for x, y in zip(xs, ys):
        danger.add((x, y))

    # Bombs about to explode: project blast radius (stops at walls)
    for (bx, by), timer in bombs:
        danger.add((bx, by))
        for dx, dy in DIRECTIONS:
            for step in range(1, 4):  # blast reaches 3 tiles
                nx, ny = bx + dx * step, by + dy * step
                if not (0 <= nx < width and 0 <= ny < height):
                    break
                if field[nx, ny] == -1:  # stone wall blocks it
                    break
                danger.add((nx, ny))
    return danger


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

    # 1-4: can I move in each direction? (not wall, not crate, not bomb, not occupied)
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

    # 11-14: one-hot direction toward the nearest coin (all zero if none)
    coin_dir = [0.0, 0.0, 0.0, 0.0]
    if coins:
        nearest = min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        cx, cy = nearest[0] - x, nearest[1] - y
        if abs(cx) > abs(cy):
            coin_dir[1 if cx > 0 else 3] = 1.0  # RIGHT or LEFT
        elif cy != 0:
            coin_dir[2 if cy > 0 else 0] = 1.0  # DOWN or UP
    features.extend(coin_dir)

    return np.array(features, dtype=np.float32)