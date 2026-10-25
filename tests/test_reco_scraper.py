from pathlib import Path

import pandas as pd

from cubesight_ml.datasets.reco_scraper import (
    RecoScraper,
    RecoSolve,
    classify_cfop_stage,
    compute_move_frequencies,
    get_move_category,
    load_reconstructions,
    normalize_cube_move,
    split_reconstructions,
    tokenize_moves,
    verify_solve_reconstruction,
)


def test_normalize_cube_move():
    assert normalize_cube_move("R") == "R"
    assert normalize_cube_move("U'") == "U'"
    assert normalize_cube_move("U2") == "U2"
    # Double prime turns
    assert normalize_cube_move("U2'") == "U2"
    assert normalize_cube_move("R2'") == "R2"
    assert normalize_cube_move("L2'") == "L2"
    assert normalize_cube_move("r2'") == "r2"
    # Triple turns
    assert normalize_cube_move("R3") == "R'"
    assert normalize_cube_move("U3") == "U'"
    # Wide moves
    assert normalize_cube_move("Rw") == "r"
    assert normalize_cube_move("Rw'") == "r'"
    assert normalize_cube_move("Rw2") == "r2"
    assert normalize_cube_move("2R") == "r"
    assert normalize_cube_move("2R'") == "r'"
    assert normalize_cube_move("Lw") == "l"
    # Punctuation and quotes
    assert normalize_cube_move("(R") == "R"
    assert normalize_cube_move("U')") == "U'"
    assert normalize_cube_move("U’") == "U'"  # noqa: RUF001


def test_tokenize_moves():
    tokens = tokenize_moves("x' z' (r' F R2) r' U r D F2 U2' R3")
    assert tokens == [
        "x'",
        "z'",
        "r'",
        "F",
        "R2",
        "r'",
        "U",
        "r",
        "D",
        "F2",
        "U2",
        "R'",
    ]


def test_classify_cfop_stage():
    assert classify_cfop_stage("inspection") == "inspection"
    assert classify_cfop_stage("cross") == "cross"
    assert classify_cfop_stage("xcross") == "xcross"
    assert classify_cfop_stage("xxcross") == "xxcross"
    assert classify_cfop_stage("1st pair") == "f2l_1"
    assert classify_cfop_stage("pair 2") == "f2l_2"
    assert classify_cfop_stage("3rd pair") == "f2l_3"
    assert classify_cfop_stage("4th pair / ZBLS") == "f2l_4"
    assert classify_cfop_stage("OLL(CP)") == "oll"
    assert classify_cfop_stage("COLL") == "oll"
    assert classify_cfop_stage("PLL") == "pll"
    assert classify_cfop_stage("AUF") == "auf"
    assert classify_cfop_stage("ZBLL") == "zbll"
    assert classify_cfop_stage("something unknown") == "other"


def test_verify_solve_reconstruction():
    scramble = "U' L2 U L2 B2 D' U' B2 L2 U' F' L2 D R2 U L B' D2 F2 D2 R'"
    solution = [
        "x'",
        "z'",  # inspection
        "r'",
        "F",
        "R2",
        "r'",
        "U",
        "r",
        "D",
        "F2",
        "U'",
        "U'",
        "U'",
        "R'",
        "U",
        "R",
        "U'",
        "R",
        "U",
        "R'",
        "R",
        "U",
        "R'",
        "U",
        "R",
        "U2'",
        "R'",
        "U",
    ]
    # Valid solve
    assert verify_solve_reconstruction(scramble, solution) is True

    # Tampered solution (missing last AUF)
    assert verify_solve_reconstruction(scramble, solution[:-1]) is False

    # Empty scramble
    assert verify_solve_reconstruction("", solution) is False


def test_reco_solve_flat_dict():
    solve = RecoSolve(
        solve_id=14241,
        puzzle="3x3",
        method="CFOP",
        solver="Yang Pin Xiu",
        result_time=2.74,
        date="2026-10-06",
        competition="Guangzhou Grand Open 2026",
        tags=["NR"],
        movecount=26,
        tps=9.49,
        reconstructed_by="Stewy",
        scramble="U' L2 U L2 B2 D' U' B2 L2 U' F' L2 D R2 U L B' D2 F2 D2 R'",
        scramble_tokens=["U'", "L2"],
        solution_raw="r' F R2 // cross\nU // AUF",
        solution_moves=["r'", "F", "R2", "U"],
        stage_moves={"cross": "r' F R2", "auf": "U"},
        stage_movecounts={"cross": 3, "auf": 1},
        stats={
            "Time": {"Total": "2.74", "F2L": "1.93"},
            "STM": {"Total": "26", "F2L": "18"},
        },
        url="https://reco.nz/solve/14241",
        is_valid_solution=True,
    )
    flat = solve.to_flat_dict()
    assert flat["solve_id"] == 14241
    assert flat["solver"] == "Yang Pin Xiu"
    assert flat["result_time"] == 2.74
    assert flat["is_valid_solution"] is True
    assert flat["cross_moves"] == "r' F R2"
    assert flat["stat_time_total"] == 2.74
    assert flat["stat_stm_total"] == 26


def test_save_and_load_reconstructions(tmp_path: Path):
    solve = RecoSolve(
        solve_id=101,
        puzzle="3x3",
        method="CFOP",
        solver="Tester",
        result_time=5.55,
        date="2026-01-01",
        competition="Test Open 2026",
        tags=["PB"],
        movecount=50,
        tps=9.01,
        reconstructed_by="Recon",
        scramble="R U R' U'",
        scramble_tokens=["R", "U", "R'", "U'"],
        solution_raw="U R U' R'",
        solution_moves=["U", "R", "U'", "R'"],
        stage_moves={"cross": "U R", "pll": "U' R'"},
        stage_movecounts={"cross": 2, "pll": 2},
        stats={},
        url="https://reco.nz/solve/101",
        is_valid_solution=True,
    )

    jsonl_path = tmp_path / "solves.jsonl"
    RecoScraper.save_jsonl([solve], jsonl_path)
    assert jsonl_path.exists()

    df = load_reconstructions(jsonl_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["solve_id"] == 101
    assert df.iloc[0]["solver"] == "Tester"
    assert df.iloc[0]["result_time"] == 5.55

    csv_path = tmp_path / "solves.csv"
    RecoScraper.save_csv([solve], csv_path)
    assert csv_path.exists()
    df_csv = load_reconstructions(csv_path)
    assert len(df_csv) == 1
    assert df_csv.iloc[0]["solve_id"] == 101


def test_split_reconstructions(tmp_path: Path):
    solves = [
        RecoSolve(
            solve_id=i,
            puzzle="3x3",
            method="CFOP",
            solver=f"Solver_{i}",
            result_time=5.0 + i,
            date="2026-01-01",
            competition="Test Open",
            tags=[],
            movecount=50,
            tps=10.0,
            reconstructed_by="Recon",
            scramble="R U R' U'",
            scramble_tokens=["R", "U", "R'", "U'"],
            solution_raw="U R U' R'",
            solution_moves=["U", "R", "U'", "R'"],
            stage_moves={},
            stage_movecounts={},
            stats={},
            url=f"https://reco.nz/solve/{i}",
            is_valid_solution=True,
        )
        for i in range(1, 7)
    ]

    src_jsonl = tmp_path / "all.jsonl"
    src_csv = tmp_path / "all.csv"
    RecoScraper.save_jsonl(solves, src_jsonl)
    RecoScraper.save_csv(solves, src_csv)

    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"

    paths, val_df, test_df = split_reconstructions(
        input_jsonl=src_jsonl,
        input_csv=src_csv,
        val_dir=val_dir,
        test_dir=test_dir,
        val_ratio=2.0 / 3.0,
        seed=42,
    )

    assert len(val_df) == 4
    assert len(test_df) == 2
    assert paths["val_jsonl"].exists()
    assert paths["val_csv"].exists()
    assert paths["test_jsonl"].exists()
    assert paths["test_csv"].exists()

    # All solve IDs should be preserved without duplication
    all_split_ids = sorted(list(val_df["solve_id"]) + list(test_df["solve_id"]))
    assert all_split_ids == [1, 2, 3, 4, 5, 6]

    # Verify JSONL content matches DataFrame
    val_jsonl_df = load_reconstructions(paths["val_jsonl"])
    assert list(val_jsonl_df["solve_id"]) == list(val_df["solve_id"])
    test_jsonl_df = load_reconstructions(paths["test_jsonl"])
    assert list(test_jsonl_df["solve_id"]) == list(test_df["solve_id"])


def test_compute_move_frequencies(tmp_path: Path):
    assert get_move_category("U") == "Face"
    assert get_move_category("M") == "Slice"
    assert get_move_category("r") == "Wide"
    assert get_move_category("y") == "Rotation"

    solve1 = RecoSolve(
        solve_id=1,
        puzzle="3x3",
        method="CFOP",
        solver="A",
        result_time=5.0,
        date="2026-01-01",
        competition="Open",
        tags=[],
        movecount=4,
        tps=10.0,
        reconstructed_by="Recon",
        scramble="R U R' U'",
        scramble_tokens=["R", "U", "R'", "U'"],
        solution_raw="R U R' U'",
        solution_moves=["R", "U", "R'", "U'"],
        stage_moves={"inspection": "x2 y"},
        stage_movecounts={},
        stats={},
        url="https://reco.nz/solve/1",
        is_valid_solution=True,
    )
    solve2 = RecoSolve(
        solve_id=2,
        puzzle="3x3",
        method="CFOP",
        solver="B",
        result_time=6.0,
        date="2026-01-01",
        competition="Open",
        tags=[],
        movecount=2,
        tps=8.0,
        reconstructed_by="Recon",
        scramble="R U R' U'",
        scramble_tokens=["R", "U", "R'", "U'"],
        solution_raw="R R",
        solution_moves=["R", "R"],
        stage_moves={"inspection": "z"},
        stage_movecounts={},
        stats={},
        url="https://reco.nz/solve/2",
        is_valid_solution=False,
    )

    jsonl_path = tmp_path / "test_solves.jsonl"
    RecoScraper.save_jsonl([solve1, solve2], jsonl_path)

    # 1. Without inspection, all solves:
    # Moves in solutions: R (3), U (1), R' (1), U' (1) -> total 6
    df = compute_move_frequencies(jsonl_path, include_inspection=False)
    assert len(df) == 54  # All 54 moves present
    assert df["count"].sum() == 6
    assert abs(df["percentage"].sum() - 100.0) < 1e-4

    r_row = df[df["move"] == "R"].iloc[0]
    assert r_row["count"] == 3
    assert abs(r_row["percentage"] - 50.0) < 1e-4

    # 2. Valid only: only solve 1 (4 moves: R, U, R', U' -> 25% each)
    df_valid = compute_move_frequencies(jsonl_path, valid_only=True)
    assert df_valid["count"].sum() == 4
    r_val = df_valid[df_valid["move"] == "R"].iloc[0]
    assert r_val["count"] == 1
    assert abs(r_val["percentage"] - 25.0) < 1e-4

    # 3. With inspection on solve 1 and 2:
    # 6 moves + inspection (x2, y, z = 3 moves) -> total 9 moves
    df_insp = compute_move_frequencies(jsonl_path, include_inspection=True)
    assert df_insp["count"].sum() == 9
    x2_row = df_insp[df_insp["move"] == "x2"].iloc[0]
    assert x2_row["count"] == 1
