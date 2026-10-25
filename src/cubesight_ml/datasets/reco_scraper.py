"""Module for scraping, parsing, normalizing, verifying, and exporting CFOP 3x3
Rubik's Cube reconstructions from https://reco.nz/ for data analysis and ML.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import certifi
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cubesight_ml.cube.moves import (
    ALL_MOVES,
    CORNERS_FACES,
    EDGES_FACES,
    FACE_MOVES,
    MOVE_TO_INDEX,
    ROTATION_MOVES,
    SLICE_MOVES,
    STICKER_MOVE_PERMS_54,
    WIDE_MOVES,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

BASE_URL = "https://reco.nz"
INDEX_URL = f"{BASE_URL}/solve/index"
SOLVE_URL = f"{BASE_URL}/solve"

# Mapping to canonical move names
MOVE_SYNONYMS: dict[str, str] = {
    # Wide moves
    "Rw": "r",
    "Rw'": "r'",
    "Rw2": "r2",
    "2R": "r",
    "2R'": "r'",
    "2R2": "r2",
    "Lw": "l",
    "Lw'": "l'",
    "Lw2": "l2",
    "2L": "l",
    "2L'": "l'",
    "2L2": "l2",
    "Fw": "f",
    "Fw'": "f'",
    "Fw2": "f2",
    "2F": "f",
    "2F'": "f'",
    "2F2": "f2",
    "Bw": "b",
    "Bw'": "b'",
    "Bw2": "b2",
    "2B": "b",
    "2B'": "b'",
    "2B2": "b2",
    "Uw": "u",
    "Uw'": "u'",
    "Uw2": "u2",
    "2U": "u",
    "2U'": "u'",
    "2U2": "u2",
    "Dw": "d",
    "Dw'": "d'",
    "Dw2": "d2",
    "2D": "d",
    "2D'": "d'",
    "2D2": "d2",
}

VALID_MOVE_SET = set(ALL_MOVES)


def normalize_cube_move(token: str) -> str:
    """Normalizes a raw move string into canonical WCA / STM notation.

    Removes parentheses, handles typographical quotes, normalizes 2' -> 2,
    3 -> ', and wide moves (Rw -> r).
    """
    token = token.strip().strip("(),[]{}")
    if not token:
        return ""

    # Replace typographical apostrophes
    token = (
        token.replace("’", "'").replace("‘", "'").replace("`", "'")  # noqa: RUF001
    )

    # Double turn prime: R2' -> R2, U2' -> U2, etc.
    if token.endswith("2'"):
        token = token[:-2] + "2"

    # Triple turns: R3 -> R', U3 -> U'
    if token.endswith("3"):
        token = token[:-1] + "'"

    # Check synonyms (e.g. Rw -> r, 2R -> r)
    if token in MOVE_SYNONYMS:
        token = MOVE_SYNONYMS[token]

    return token


def tokenize_moves(move_str: str) -> list[str]:
    """Splits a move string into normalized valid cube moves."""
    tokens: list[str] = []
    for raw in move_str.split():
        norm = normalize_cube_move(raw)
        if norm and norm in VALID_MOVE_SET:
            tokens.append(norm)
    return tokens


def classify_cfop_stage(comment: str) -> str:
    """Classifies a reconstruction step comment into a standard CFOP stage.

    Returns one of:
    'inspection', 'cross', 'xcross', 'xxcross', 'f2l_1', 'f2l_2', 'f2l_3', 'f2l_4',
    'oll', 'pll', 'auf', 'other'
    """
    c = comment.lower().strip()
    if not c:
        return "other"

    if "inspection" in c:
        return "inspection"

    if "xxcross" in c or "xx-cross" in c or "double xcross" in c:
        return "xxcross"
    if "xcross" in c or "x-cross" in c:
        return "xcross"
    if "cross" in c:
        return "cross"

    if any(k in c for k in ["1st pair", "pair 1", "first pair", "p1"]):
        return "f2l_1"
    if any(k in c for k in ["2nd pair", "pair 2", "second pair", "p2"]):
        return "f2l_2"
    if any(k in c for k in ["3rd pair", "pair 3", "third pair", "p3"]):
        return "f2l_3"
    if any(
        k in c
        for k in [
            "4th pair",
            "pair 4",
            "fourth pair",
            "p4",
            "last slot",
            "ls",
            "zbls",
            "vls",
            "cls",
        ]
    ):
        return "f2l_4"

    if "zbll" in c:
        return "zbll"
    if any(k in c for k in ["oll", "coll", "winter variation", "wv"]):
        return "oll"
    if any(k in c for k in ["pll", "epll", "cpll"]):
        return "pll"

    if "auf" in c or c in ("auf", "u", "u'", "u2"):
        return "auf"

    return "other"


def verify_solve_reconstruction(
    scramble: str, solution_moves: list[str]
) -> bool:
    """Verifies whether applying the scramble followed by the solution moves

    results in a monochromatic 6-face solved cube using STICKER_MOVE_PERMS_54.
    """
    scramble_tokens = tokenize_moves(scramble)
    if not scramble_tokens:
        return False

    sol_tokens: list[str] = []
    for m in solution_moves:
        norm = normalize_cube_move(m)
        if norm:
            if norm not in VALID_MOVE_SET:
                return False
            sol_tokens.append(norm)

    # Initialize 54 stickers:
    # 0..5: centers (0=U, 1=R, 2=F, 3=D, 4=L, 5=B)
    # 6..29: corners
    # 30..53: edges
    colors = np.zeros(54, dtype=np.int8)
    for c in range(6):
        colors[c] = c
    for s in range(8):
        for j in range(3):
            colors[6 + 3 * s + j] = CORNERS_FACES[s][j]
    for s in range(12):
        for j in range(2):
            colors[30 + 2 * s + j] = EDGES_FACES[s][j]

    # Apply scramble
    for m in scramble_tokens:
        idx = MOVE_TO_INDEX.get(m)
        if idx is None:
            return False
        colors = colors[STICKER_MOVE_PERMS_54[idx]]

    # Apply solution
    for m in sol_tokens:
        idx = MOVE_TO_INDEX.get(m)
        if idx is None:
            return False
        colors = colors[STICKER_MOVE_PERMS_54[idx]]

    # Check each face has 9 stickers matching its center
    center_colors = colors[:6]
    for s in range(8):
        for j in range(3):
            face = CORNERS_FACES[s][j]
            if colors[6 + 3 * s + j] != center_colors[face]:
                return False

    for s in range(12):
        for j in range(2):
            face = EDGES_FACES[s][j]
            if colors[30 + 2 * s + j] != center_colors[face]:
                return False

    return True


@dataclass
class RecoStep:
    stage: str
    moves: str
    tokens: list[str]
    comment: str
    raw_line: str


@dataclass
class RecoSolve:
    solve_id: int
    puzzle: str
    method: str
    solver: str
    result_time: float | None
    date: str
    competition: str
    tags: list[str]
    movecount: int | None
    tps: float | None
    reconstructed_by: str
    scramble: str
    scramble_tokens: list[str]
    solution_raw: str
    solution_moves: list[str]
    steps: list[RecoStep] = field(default_factory=list)
    stage_moves: dict[str, str] = field(default_factory=dict)
    stage_movecounts: dict[str, int] = field(default_factory=dict)
    stats: dict[str, dict[str, str]] = field(default_factory=dict)
    url: str = ""
    alg_cubing_url: str | None = None
    video_url: str | None = None
    is_valid_solution: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts to a JSON-serializable dictionary."""
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flattens solve attributes into a single-level dictionary ideal for

        pandas DataFrame or CSV data analysis.
        """
        flat: dict[str, Any] = {
            "solve_id": self.solve_id,
            "puzzle": self.puzzle,
            "method": self.method,
            "solver": self.solver,
            "result_time": self.result_time,
            "date": self.date,
            "competition": self.competition,
            "tags": ";".join(self.tags),
            "movecount": self.movecount,
            "tps": self.tps,
            "reconstructed_by": self.reconstructed_by,
            "is_valid_solution": self.is_valid_solution,
            "scramble": self.scramble,
            "scramble_length": len(self.scramble_tokens),
            "solution_length": len(self.solution_moves),
            "solution_moves": " ".join(self.solution_moves),
            "inspection": self.stage_moves.get("inspection", ""),
            "cross_moves": self.stage_moves.get("cross", ""),
            "cross_movecount": self.stage_movecounts.get("cross", 0),
            "f2l_1_moves": self.stage_moves.get("f2l_1", ""),
            "f2l_2_moves": self.stage_moves.get("f2l_2", ""),
            "f2l_3_moves": self.stage_moves.get("f2l_3", ""),
            "f2l_4_moves": self.stage_moves.get("f2l_4", ""),
            "f2l_total_moves": self.stage_moves.get("f2l_total", ""),
            "f2l_total_movecount": self.stage_movecounts.get("f2l_total", 0),
            "oll_moves": self.stage_moves.get("oll", ""),
            "oll_movecount": self.stage_movecounts.get("oll", 0),
            "pll_moves": self.stage_moves.get("pll", ""),
            "pll_movecount": self.stage_movecounts.get("pll", 0),
            "auf_moves": self.stage_moves.get("auf", ""),
            "auf_movecount": self.stage_movecounts.get("auf", 0),
        }

        # Add solvestats if available
        # Time, STM, STPS for Total, F2L, LL, Cross+1
        for metric in ["Time", "STM", "STPS", "ETM", "ETPS", "Split"]:
            m_dict = self.stats.get(metric, {})
            for col in ["Total", "F2L", "LL", "Cross+1", "OLS", "PLL"]:
                val = m_dict.get(col)
                key = f"stat_{metric.lower()}_{col.lower().replace('+', '_')}"
                if val is not None:
                    # Clean % or convert numeric
                    clean_val = val.rstrip("%")
                    try:
                        flat[key] = (
                            float(clean_val)
                            if "." in clean_val
                            else int(clean_val)
                        )
                    except ValueError:
                        flat[key] = val
                else:
                    flat[key] = None

        flat["url"] = self.url
        flat["alg_cubing_url"] = self.alg_cubing_url
        flat["video_url"] = self.video_url
        return flat


class RecoScraper:
    """Robust scraper for https://reco.nz/ with polite rate limiting, retry

    handling, CFOP step parsing, and cube verification.
    """

    def __init__(
        self,
        delay: float = 0.25,
        timeout: int = 15,
        verify_ssl: bool | str | None = None,
    ) -> None:
        self.delay = delay
        self.timeout = timeout
        self.verify_ssl = certifi.where() if verify_ssl is None else verify_ssl

        # Configure session with retries and realistic headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get(self, url: str, params: dict | None = None) -> str:
        """Fetches page content with polite delay and error handling."""
        if self.delay > 0:
            time.sleep(self.delay)
        resp = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text

    def fetch_index_page(
        self,
        page: int = 1,
        puzzle: str = "3x3",
        method: str = "CFOP",
        sort: str = "id_desc",
        solvers: list[str] | str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetches a single page of solves from the listing index.

        Returns (solve_rows, has_next_page).
        """
        params: dict[str, Any] = {
            "sort": sort,
            "page": page,
        }
        if puzzle:
            params[puzzle] = "on"
        if method:
            params[method] = "on"
        if solvers:
            params["solver[]"] = (
                solvers if isinstance(solvers, list) else [solvers]
            )

        html = self._get(INDEX_URL, params=params)
        soup = BeautifulSoup(html, "html.parser")

        rows = soup.select("tr.solve-row")
        solves_meta: list[dict[str, Any]] = []

        for r in rows:
            sid_str = r.get("data-id")
            if not sid_str:
                continue
            cells = [td.get_text(strip=True) for td in r.find_all("td")]
            # Format: [ID, Puzzle, Result, Solver, Method, Date, Competition, Tags, Movecount, TPS, Reconstructed By]
            if len(cells) >= 11:
                res_float = None
                with contextlib.suppress(ValueError, TypeError):
                    res_float = float(cells[2])

                mc_int = None
                with contextlib.suppress(ValueError, TypeError):
                    mc_int = int(cells[8])

                tps_float = None
                with contextlib.suppress(ValueError, TypeError):
                    tps_float = float(cells[9])

                tags_list = [
                    t.strip("[]")
                    for t in re.findall(r"\[.*?\]", cells[7])
                    if t.strip("[]")
                ]
                if not tags_list and cells[7].strip():
                    tags_list = [cells[7].strip()]

                solves_meta.append(
                    {
                        "solve_id": int(sid_str),
                        "puzzle": cells[1],
                        "result_time": res_float,
                        "solver": cells[3],
                        "method": cells[4],
                        "date": cells[5],
                        "competition": cells[6],
                        "tags": tags_list,
                        "movecount": mc_int,
                        "tps": tps_float,
                        "reconstructed_by": cells[10],
                    }
                )

        # Check pagination for 'Next'
        pagination = soup.select(".pagination a")
        has_next = any("Next" in a.get_text() for a in pagination)

        return solves_meta, has_next

    def fetch_solve_detail(
        self,
        solve_id: int,
        summary_meta: dict[str, Any] | None = None,
        verify: bool = True,
    ) -> RecoSolve | None:
        """Fetches and parses a single reconstruction page by solve_id."""
        url = f"{SOLVE_URL}/{solve_id}"
        try:
            html = self._get(url)
        except (requests.RequestException, OSError) as e:
            logger.warning("Failed to fetch solve %d: %s", solve_id, e)
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Fallback metadata from H1 and H3 if summary_meta not provided
        h1 = soup.select_one("h1")
        h3 = soup.select_one("h3")

        solver = ""
        result_time = None
        puzzle = "3x3"
        method = "CFOP"
        date = ""
        competition = ""
        reconstructed_by = ""
        tags: list[str] = []
        movecount = None
        tps = None

        if summary_meta:
            solver = summary_meta.get("solver", "")
            result_time = summary_meta.get("result_time")
            puzzle = summary_meta.get("puzzle", "3x3")
            method = summary_meta.get("method", "CFOP")
            date = summary_meta.get("date", "")
            competition = summary_meta.get("competition", "")
            reconstructed_by = summary_meta.get("reconstructed_by", "")
            tags = summary_meta.get("tags", [])
            movecount = summary_meta.get("movecount")
            tps = summary_meta.get("tps")
        else:
            if h1:
                h1_text = h1.get_text(" ", strip=True)
                match = re.search(
                    r"^(.*?)\s*-\s*([\d\.]+)?\s*(\S+)\s*solve", h1_text
                )
                if match:
                    solver = match.group(1).strip()
                    with contextlib.suppress(ValueError, TypeError):
                        result_time = float(match.group(2))
                    puzzle = match.group(3).strip()
                else:
                    solver = h1_text

            if h3:
                h3_text = h3.get_text(" ", strip=True)
                tags = [
                    t.strip("[]")
                    for t in re.findall(r"\[.*?\]", h3_text)
                    if t.strip("[]")
                ]
                clean_h3 = re.sub(r"\[.*?\]", "", h3_text).strip()
                parts = [p.strip() for p in clean_h3.split("-")]
                if parts:
                    date = parts[0]
                if len(parts) > 1:
                    competition = parts[1]
                recon_link = soup.select_one("#reconstructor-link")
                if recon_link:
                    reconstructed_by = recon_link.get_text(strip=True)

        # Scramble and Solution extraction
        alg_link = soup.find("a", href=re.compile(r"alg\.cubing\.net"))
        scramble = ""
        solution_raw = ""
        alg_cubing_url = None

        if alg_link:
            alg_cubing_url = alg_link.get("href")
            try:
                parsed = parse_qs(urlparse(alg_cubing_url).query)
                scramble = parsed.get("setup", [""])[0].strip()
                solution_raw = parsed.get("alg", [""])[0].strip()
            except (ValueError, KeyError) as e:
                logger.debug(
                    "Error parsing alg.cubing.net query for solve %d: %s",
                    solve_id,
                    e,
                )

        # Fallback to #reconstruction div if alg.cubing.net was missing
        reco_div = soup.select_one("#reconstruction")
        if not scramble and reco_div:
            raw_div_text = reco_div.get_text("\n", strip=True)
            lines = [
                line_text.strip()
                for line_text in raw_div_text.splitlines()
                if line_text.strip()
            ]
            if lines:
                scramble = lines[0]
                if not solution_raw and len(lines) > 1:
                    solution_raw = "\n".join(lines[1:])

        # Parse solution steps and CFOP breakdown
        steps: list[RecoStep] = []
        solution_moves: list[str] = []
        stage_moves: dict[str, list[str]] = {
            "inspection": [],
            "cross": [],
            "f2l_1": [],
            "f2l_2": [],
            "f2l_3": [],
            "f2l_4": [],
            "f2l_total": [],
            "oll": [],
            "pll": [],
            "auf": [],
            "other": [],
        }

        for line in solution_raw.splitlines():
            line = line.strip()
            if not line:
                continue

            comment = ""
            if "//" in line:
                move_part, comment = line.split("//", 1)
                move_part = move_part.strip()
                comment = comment.strip()
            else:
                move_part = line

            tokens = tokenize_moves(move_part)
            stage = classify_cfop_stage(comment)

            steps.append(
                RecoStep(
                    stage=stage,
                    moves=move_part,
                    tokens=tokens,
                    comment=comment,
                    raw_line=line,
                )
            )

            # Route tokens to stages
            if stage == "inspection":
                stage_moves["inspection"].extend(tokens)
            else:
                solution_moves.extend(tokens)
                if stage in ("cross", "xcross", "xxcross"):
                    stage_moves["cross"].extend(tokens)
                    if stage == "xcross":
                        stage_moves["f2l_1"].extend(tokens)
                    elif stage == "xxcross":
                        stage_moves["f2l_1"].extend(tokens)
                        stage_moves["f2l_2"].extend(tokens)
                elif stage in ("f2l_1", "f2l_2", "f2l_3", "f2l_4"):
                    stage_moves[stage].extend(tokens)
                    stage_moves["f2l_total"].extend(tokens)
                elif stage in ("oll", "zbll"):
                    stage_moves["oll"].extend(tokens)
                elif stage == "pll":
                    stage_moves["pll"].extend(tokens)
                elif stage == "auf":
                    stage_moves["auf"].extend(tokens)
                else:
                    stage_moves["other"].extend(tokens)

        # Build collapsed stage strings and counts
        flat_stage_moves = {
            k: " ".join(toks) for k, toks in stage_moves.items() if toks
        }
        flat_stage_counts = {
            k: len(toks) for k, toks in stage_moves.items() if toks
        }

        # Video URL
        iframe = soup.select_one("iframe")
        video_url = iframe.get("src") if iframe and iframe.get("src") else None

        # Solvestats table
        stats: dict[str, dict[str, str]] = {}
        stats_table = soup.select_one("#solvestats")
        if stats_table:
            first_tr = stats_table.select_one("tr")
            if first_tr:
                cols = [
                    th.get_text(strip=True)
                    for th in first_tr.select("th")
                    if th.get_text(strip=True)
                ]
                for tr in stats_table.select("tr")[1:]:
                    hdr = tr.select_one("th")
                    if not hdr:
                        continue
                    metric_name = hdr.get_text(strip=True)
                    cell_vals = [
                        td.get_text(strip=True) for td in tr.select("td")
                    ]
                    stats[metric_name] = dict(
                        zip(cols, cell_vals, strict=False)
                    )

        scramble_tokens = tokenize_moves(scramble)

        # Verification with simulator
        is_valid = None
        if verify and scramble and solution_moves:
            try:
                # Include inspection rotations if any
                all_sim_moves = (
                    stage_moves.get("inspection", []) + solution_moves
                )
                is_valid = verify_solve_reconstruction(scramble, all_sim_moves)
            except (ValueError, KeyError, IndexError) as e:
                logger.debug(
                    "Verification exception for solve %d: %s", solve_id, e
                )
                is_valid = False

        return RecoSolve(
            solve_id=solve_id,
            puzzle=puzzle,
            method=method,
            solver=solver,
            result_time=result_time,
            date=date,
            competition=competition,
            tags=tags,
            movecount=movecount or len(solution_moves),
            tps=tps,
            reconstructed_by=reconstructed_by,
            scramble=scramble,
            scramble_tokens=scramble_tokens,
            solution_raw=solution_raw,
            solution_moves=solution_moves,
            steps=steps,
            stage_moves=flat_stage_moves,
            stage_movecounts=flat_stage_counts,
            stats=stats,
            url=url,
            alg_cubing_url=alg_cubing_url,
            video_url=video_url,
            is_valid_solution=is_valid,
        )

    def scrape(
        self,
        limit: int | None = None,
        max_pages: int | None = None,
        start_page: int = 1,
        puzzle: str = "3x3",
        method: str = "CFOP",
        sort: str = "id_desc",
        solvers: list[str] | str | None = None,
        concurrency: int = 1,
        verify: bool = True,
        existing_ids: set[int] | None = None,
        callback: Callable[[RecoSolve], None] | None = None,
    ) -> Iterator[RecoSolve]:
        """Iteratively fetches solve metadata and details across index pages.

        Yields RecoSolve objects.
        """
        existing = existing_ids or set()
        count = 0
        current_page = start_page

        while True:
            if max_pages and current_page >= start_page + max_pages:
                break

            logger.info(
                "Fetching index page %d (%s, %s)...",
                current_page,
                puzzle,
                method,
            )
            solves_meta, has_next = self.fetch_index_page(
                page=current_page,
                puzzle=puzzle,
                method=method,
                sort=sort,
                solvers=solvers,
            )

            if not solves_meta:
                break

            to_fetch = [m for m in solves_meta if m["solve_id"] not in existing]
            if concurrency > 1 and len(to_fetch) > 1:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            self.fetch_solve_detail,
                            m["solve_id"],
                            m,
                            verify,
                        )
                        for m in to_fetch
                    ]
                    for fut in as_completed(futures):
                        solve = fut.result()
                        if solve is not None:
                            if callback:
                                callback(solve)
                            yield solve
                            count += 1
                            existing.add(solve.solve_id)
                        if limit and count >= limit:
                            return
            else:
                for meta in to_fetch:
                    sid = meta["solve_id"]
                    solve = self.fetch_solve_detail(
                        sid, summary_meta=meta, verify=verify
                    )
                    if solve is not None:
                        if callback:
                            callback(solve)
                        yield solve
                        count += 1
                        existing.add(sid)

                    if limit and count >= limit:
                        return

            if not has_next:
                break

            current_page += 1

    @staticmethod
    def save_jsonl(
        solves: Iterator[RecoSolve] | list[RecoSolve],
        filepath: Path | str,
        append: bool = False,
    ) -> int:
        """Saves solves to a newline-delimited JSON (JSONL) file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        written = 0
        with path.open(mode, encoding="utf-8") as f:
            for s in solves:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
                written += 1
        return written

    @staticmethod
    def save_json(
        solves: list[RecoSolve],
        filepath: Path | str,
    ) -> None:
        """Saves a list of solves as a single formatted JSON array."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in solves],
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def save_csv(
        solves: list[RecoSolve] | Iterator[RecoSolve],
        filepath: Path | str,
    ) -> int:
        """Saves flattened solve records to a CSV file for pandas / data

        analysis.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [s.to_flat_dict() for s in solves]
        if not rows:
            return 0

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)


def load_reconstructions(filepath: Path | str) -> pd.DataFrame:
    """Convenience helper to load scraped reconstructions (JSONL, JSON, or CSV)

    directly into a pandas DataFrame ready for data analysis.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if "steps" in data:
                        steps = [RecoStep(**st) for st in data.get("steps", [])]
                        data_copy = dict(data)
                        data_copy["steps"] = steps
                        solve = RecoSolve(**data_copy)
                        records.append(solve.to_flat_dict())
                    else:
                        records.append(data)
        return pd.DataFrame(records)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw_list = json.load(f)
        records = []
        for data in raw_list:
            if "steps" in data:
                steps = [RecoStep(**st) for st in data.get("steps", [])]
                data_copy = dict(data)
                data_copy["steps"] = steps
                solve = RecoSolve(**data_copy)
                records.append(solve.to_flat_dict())
            else:
                records.append(data)
        return pd.DataFrame(records)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def split_reconstructions(
    input_jsonl: str | Path,
    input_csv: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    val_ratio: float = 2.0 / 3.0,
    seed: int = 42,
    val_prefix: str = "reco_cfop_val",
    test_prefix: str = "reco_cfop_test",
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    """Split JSONL and CSV reconstructions datasets into val and test splits.

    Args:
        input_jsonl: Path to original JSONL file.
        input_csv: Path to original CSV file.
        val_dir: Destination directory for validation set.
        test_dir: Destination directory for test set.
        val_ratio: Fraction of dataset allocated to validation (default 2/3).
        seed: Random seed for deterministic shuffling.
        val_prefix: Base filename for validation files.
        test_prefix: Base filename for test files.

    Returns:
        Tuple of (dict of output file paths, val DataFrame, test DataFrame).
    """
    jsonl_path = Path(input_jsonl)
    csv_path = Path(input_csv)
    val_dir_path = Path(val_dir)
    test_dir_path = Path(test_dir)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {jsonl_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    val_dir_path.mkdir(parents=True, exist_ok=True)
    test_dir_path.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as f:
        jsonl_lines = [line.strip() for line in f if line.strip()]

    csv_df = pd.read_csv(csv_path, low_memory=False)

    n_jsonl = len(jsonl_lines)
    n_csv = len(csv_df)
    if n_jsonl != n_csv:
        raise ValueError(
            f"Row count mismatch: JSONL has {n_jsonl} lines, CSV has {n_csv} rows."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_jsonl)

    n_val = round(n_jsonl * val_ratio)
    val_indices = perm[:n_val]
    test_indices = perm[n_val:]

    val_lines = [jsonl_lines[i] for i in val_indices]
    test_lines = [jsonl_lines[i] for i in test_indices]

    val_df = csv_df.iloc[val_indices].reset_index(drop=True)
    test_df = csv_df.iloc[test_indices].reset_index(drop=True)

    val_jsonl_out = val_dir_path / f"{val_prefix}.jsonl"
    val_csv_out = val_dir_path / f"{val_prefix}.csv"
    test_jsonl_out = test_dir_path / f"{test_prefix}.jsonl"
    test_csv_out = test_dir_path / f"{test_prefix}.csv"

    with val_jsonl_out.open("w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    val_df.to_csv(val_csv_out, index=False)

    with test_jsonl_out.open("w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line + "\n")

    test_df.to_csv(test_csv_out, index=False)

    paths = {
        "val_jsonl": val_jsonl_out,
        "val_csv": val_csv_out,
        "test_jsonl": test_jsonl_out,
        "test_csv": test_csv_out,
    }
    return paths, val_df, test_df


def get_move_category(move: str) -> str:
    """Classifies an STM move into Face, Slice, Wide, or Rotation."""
    if move in FACE_MOVES:
        return "Face"
    if move in SLICE_MOVES:
        return "Slice"
    if move in WIDE_MOVES:
        return "Wide"
    if move in ROTATION_MOVES:
        return "Rotation"
    return "Other"


def compute_move_frequencies(
    dataset_path: str | Path,
    include_inspection: bool = False,
    valid_only: bool = False,
) -> pd.DataFrame:
    """Computes move frequency and percentage for all 54 STM moves in dataset solutions.

    Args:
        dataset_path: Path to dataset (.jsonl or .csv).
        include_inspection: Whether to include moves from the inspection phase.
        valid_only: Whether to only count moves from verified valid solutions.

    Returns:
        DataFrame containing columns:
        ['move_index', 'move', 'category', 'count', 'frequency', 'percentage']
        ordered by move_index matching ALL_MOVES.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    counter: Counter[str] = Counter()
    all_moves_set = set(ALL_MOVES)

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if valid_only and data.get("is_valid_solution") is not True:
                    continue
                moves = data.get("solution_moves", [])
                for m in moves:
                    if m in all_moves_set:
                        counter[m] += 1
                if include_inspection:
                    insp = data.get("stage_moves", {}).get("inspection", "")
                    insp_tokens = (
                        insp.split() if isinstance(insp, str) else insp
                    )
                    for m in insp_tokens:
                        if m in all_moves_set:
                            counter[m] += 1
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
        if valid_only and "is_valid_solution" in df.columns:
            df = df[df["is_valid_solution"].fillna(False).astype(bool)]
        for _, row in df.iterrows():
            sol_str = str(row.get("solution_moves", ""))
            if sol_str and sol_str != "nan":
                for m in sol_str.split():
                    if m in all_moves_set:
                        counter[m] += 1
            if include_inspection:
                insp_str = str(row.get("inspection", ""))
                if insp_str and insp_str != "nan":
                    for m in insp_str.split():
                        if m in all_moves_set:
                            counter[m] += 1
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    total_moves = sum(counter.values())

    records = []
    for idx, move in enumerate(ALL_MOVES):
        cnt = counter[move]
        freq = (cnt / total_moves) if total_moves > 0 else 0.0
        pct = freq * 100.0
        records.append(
            {
                "move_index": idx,
                "move": move,
                "category": get_move_category(move),
                "count": cnt,
                "frequency": freq,
                "percentage": pct,
            }
        )

    return pd.DataFrame(records)
