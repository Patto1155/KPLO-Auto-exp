"""Protected, deterministic KQK/KRK mate-in-one benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chess_agent import choose_move


def generate_suite(seed: int, count: int) -> list[tuple[str, str]]:
    """Generate legal positions having exactly one mating move."""
    rng = random.Random(seed)
    positions: list[tuple[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    while len(positions) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not generate enough unique tactical positions")
        board = chess.Board(None)
        attacker_king, defender_king, attacker = rng.sample(range(64), 3)
        board.set_piece_at(attacker_king, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(defender_king, chess.Piece(chess.KING, chess.BLACK))
        piece_type = rng.choice((chess.QUEEN, chess.ROOK))
        board.set_piece_at(attacker, chess.Piece(piece_type, chess.WHITE))
        board.turn = chess.WHITE
        if not board.is_valid() or board.is_check():
            continue
        mates = []
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                mates.append(move.uci())
            board.pop()
        fen = board.fen()
        if len(mates) == 1 and fen not in seen:
            seen.add(fen)
            positions.append((fen, mates[0]))
    return positions


def main() -> None:
    seed = int(os.environ.get("NRL_SEED", "101"))
    budget = json.loads(os.environ.get("NRL_BUDGET", "{}"))
    count = int(budget.get("positions", 64))
    suite = generate_suite(seed, count)
    correct = illegal = crashes = 0
    legal_move_total = 0
    started = time.perf_counter()
    results = []
    for index, (fen, expected) in enumerate(suite):
        board = chess.Board(fen)
        legal_moves = sorted(move.uci() for move in board.legal_moves)
        legal_move_total += len(legal_moves)
        try:
            prediction = choose_move(fen, legal_moves, seed + index)
        except Exception as error:
            prediction = None
            crashes += 1
            error_name = type(error).__name__
        else:
            error_name = None
        is_legal = prediction in legal_moves
        is_correct = is_legal and prediction == expected
        illegal += int(not is_legal)
        correct += int(is_correct)
        results.append({"index": index, "fen": fen, "prediction": prediction,
                        "correct": is_correct, "legal": is_legal, "error": error_name})
    inference_seconds = time.perf_counter() - started
    digest = hashlib.sha256("\n".join(fen for fen, _ in suite).encode()).hexdigest()
    metrics = {
        "tactical_pass_at_1": correct / count,
        "illegal_move_rate": illegal / count,
        "crash_rate": crashes / count,
        "positions": float(count),
        "mean_legal_moves": legal_move_total / count,
        "inference_seconds": inference_seconds,
    }
    print(json.dumps({"metrics": metrics, "suite_hash": digest, "episodes": results}, sort_keys=True))


if __name__ == "__main__":
    main()
