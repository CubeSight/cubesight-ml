import json
import urllib.request

from cubesight_ml.cube.layout import SOLVED_FLAT, from_flat
from cubesight_ml.visualization.backend import SolverAPI
from cubesight_ml.visualization.cube_state import (
    get_cubies_representation,
    state_to_stickers,
    stickers_to_face_dict,
)
from cubesight_ml.visualization.server import start_server


def test_cube_state_conversion():
    cp, co, ep, eo = from_flat(SOLVED_FLAT)
    stickers = state_to_stickers(cp, co, ep, eo)
    assert len(stickers) == 54

    # Centers are 0, 1, 2, 3, 4, 5
    assert stickers[0:6] == [0, 1, 2, 3, 4, 5]

    faces = stickers_to_face_dict(stickers)
    assert len(faces) == 6
    for _face, colors in faces.items():
        assert len(colors) == 9
        # In solved state, all 9 face stickers have the same color
        assert len(set(colors)) == 1

    cubies = get_cubies_representation(stickers)
    assert len(cubies) == 27
    # Check URF corner at (1, 1, 1)
    urf = next(c for c in cubies if c["x"] == 1 and c["y"] == 1 and c["z"] == 1)
    assert urf["faces"]["top"] == 0  # White
    assert urf["faces"]["right"] == 1  # Red
    assert urf["faces"]["front"] == 2  # Green
    assert urf["faces"]["bottom"] is None
    assert urf["faces"]["left"] is None
    assert urf["faces"]["back"] is None


def test_solver_api_moves_and_reset():
    api = SolverAPI(device="cpu")
    state = api.get_state()
    assert state["status"]["is_solved"] is True

    # Apply move R
    next_state = api.apply_move("R")
    assert next_state["status"]["is_solved"] is False

    # Apply R' to return to solved
    solved_state = api.apply_move("R'")
    assert solved_state["status"]["is_solved"] is True

    # Scramble and reset
    scrambled = api.scramble(depth=10)
    assert scrambled["status"]["is_solved"] is False
    reset_state = api.reset()
    assert reset_state["status"]["is_solved"] is True


def test_solver_api_preconditions():
    api = SolverAPI(device="cpu")

    # Cross Done has Cross solved
    cross_done = api.precondition("cross")
    assert cross_done["status"]["cross_solved"] is True

    # F2L Done has Cross and F2L solved
    f2l_done = api.precondition("f2l")
    assert f2l_done["status"]["cross_solved"] is True
    assert f2l_done["status"]["f2l_solved"] is True

    # OLL Done has Cross, F2L, and OLL solved
    oll_done = api.precondition("oll")
    assert oll_done["status"]["cross_solved"] is True
    assert oll_done["status"]["f2l_solved"] is True
    assert oll_done["status"]["oll_solved"] is True

    # PLL Done has entire cube solved
    pll_done = api.precondition("pll")
    assert pll_done["status"]["is_solved"] is True


def test_solver_api_single_stage_solve():
    api = SolverAPI(device="cpu")
    api.apply_move("R")

    # Solve PLL stage from 1 move away
    res = api.solve(stage="pll", beam_width=10, max_steps=5)
    assert res["success"] is True
    assert res["moves"] == ["R'"]
    assert res["stm_length"] == 1
    assert len(res["intermediate_states"]) == 2  # step 0 and step 1


def test_cube_stickers_remain_valid_after_moves():
    api = SolverAPI(device="cpu")
    moves_to_test = ["R", "U", "R'", "U'", "M", "E", "S", "r", "l2", "x", "y'"]

    for m in moves_to_test:
        state = api.apply_move(m)
        cubies = state["cubies"]
        assert len(cubies) == 27

        color_counts = {c: 0 for c in range(6)}
        for c in cubies:
            x, y, z = c["x"], c["y"], c["z"]
            faces = c["faces"]

            if x == 1:
                assert faces["right"] in range(6)
            else:
                assert faces["right"] is None

            if x == -1:
                assert faces["left"] in range(6)
            else:
                assert faces["left"] is None

            if y == 1:
                assert faces["top"] in range(6)
            else:
                assert faces["top"] is None

            if y == -1:
                assert faces["bottom"] in range(6)
            else:
                assert faces["bottom"] is None

            if z == 1:
                assert faces["front"] in range(6)
            else:
                assert faces["front"] is None

            if z == -1:
                assert faces["back"] in range(6)
            else:
                assert faces["back"] is None

            for val in faces.values():
                if val is not None:
                    color_counts[val] += 1

        for color_id in range(6):
            assert color_counts[color_id] == 9


def test_solver_api_multi_stage_solve():
    api = SolverAPI(device="cpu")
    api.apply_move("R")

    # Full pipeline solve from 1 move away
    res = api.solve(stage="all", beam_width=10, max_steps=5)
    assert "success" in res
    assert "moves" in res
    assert "stage_breakdown" in res
    assert len(res["stage_breakdown"]) == 4


def test_visualizer_server_endpoints():
    api = SolverAPI(device="cpu")
    server, url = start_server(api, host="127.0.0.1", port=0)

    try:
        # GET index.html
        with urllib.request.urlopen(f"{url}/") as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "CubeSight-ML" in content

        # GET /api/state
        with urllib.request.urlopen(f"{url}/api/state") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "stickers" in data
            assert len(data["stickers"]) == 54

        # POST /api/move
        req = urllib.request.Request(
            f"{url}/api/move",
            data=json.dumps({"move": "U"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"]["is_solved"] is False

        # POST /api/reset
        req = urllib.request.Request(
            f"{url}/api/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"]["is_solved"] is True
    finally:
        server.shutdown()


def test_solver_api_scramble_preserves_centers():
    """Verify that api.scramble preserves canonical centers."""
    import numpy as np

    api = SolverAPI(device="cpu")
    api.scramble(depth=20)
    cp, co, ep, eo, centers = api.env.get_state_with_centers()
    assert np.array_equal(centers[0], np.arange(6))


def test_solver_api_solve_cross_matches_center():
    """Verify that solved cross edges match the center piece in the visualizer."""
    api = SolverAPI(device="cpu")
    api.scramble(depth=4)

    res = api.solve(stage="cross", beam_width=50, max_steps=20)
    assert res["success"] is True

    # Inspect last state
    last_state = res["intermediate_states"][-1]
    faces = last_state["faces"]

    matched_cross_face = None
    for f_name, grid in faces.items():
        center = grid[4]
        cross_edges = [grid[1], grid[3], grid[5], grid[7]]
        if all(e == center for e in cross_edges):
            matched_cross_face = f_name
            break

    assert matched_cross_face is not None, (
        "Solved cross edges do not match the face's center piece!"
    )


def test_solver_api_solve_rotated_cube():
    """Verify that visualizer solve reorients rotated cube before solving."""
    import numpy as np

    api = SolverAPI(device="cpu")
    api.apply_move("x")
    _, _, _, _, centers = api.env.get_state_with_centers()
    assert not np.array_equal(centers[0], np.arange(6))

    res = api.solve(stage="cross", beam_width=50, max_steps=20)
    assert res["success"] is True
    # The first move should be the restoring rotation x'
    assert res["moves"][0] == "x'"
