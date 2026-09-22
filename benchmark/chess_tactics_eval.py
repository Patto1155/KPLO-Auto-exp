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


def forces_mate(board: chess.Board, plies: int) -> bool:
    if board.is_checkmate():
        return True
    if plies == 0 or board.is_stalemate() or board.is_insufficient_material():
        return False
    outcomes = []
    for move in list(board.legal_moves):
        board.push(move)
        outcomes.append(forces_mate(board, plies - 1))
        board.pop()
    return any(outcomes) if board.turn == chess.WHITE else bool(outcomes) and all(outcomes)


def forcing_moves(board: chess.Board, plies: int) -> list[str]:
    moves = []
    for move in list(board.legal_moves):
        board.push(move)
        wins = forces_mate(board, plies - 1)
        board.pop()
        if wins:
            moves.append(move.uci())
    return moves


def generate_mate_one(seed: int, count: int) -> list[tuple[str, str, int]]:
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
            positions.append((fen, mates[0], 1))
    return positions


def _transform_square(square: int, transform: int) -> int:
    file, rank = chess.square_file(square), chess.square_rank(square)
    if transform & 1:
        file = 7 - file
    if transform & 2:
        rank = 7 - rank
    return chess.square(file, rank)


def _transform_position(fen: str, move_text: str, transform: int, depth: int) -> tuple[str, str, int]:
    source = chess.Board(fen)
    target = chess.Board(None)
    for square, piece in source.piece_map().items():
        target.set_piece_at(_transform_square(square, transform), piece)
    target.turn = source.turn
    move = chess.Move.from_uci(move_text)
    mapped = chess.Move(_transform_square(move.from_square, transform), _transform_square(move.to_square, transform))
    return target.fen(), mapped.uci(), depth


def generate_suite(seed: int, count: int) -> list[tuple[str, str, int]]:
    mate_one_count = count // 3
    mate_two_count = count // 3
    positions = generate_mate_one(seed, mate_one_count)
    for filename, depth, amount in (
        ("chess_mate_two.json", 3, mate_two_count),
        ("chess_mate_three.json", 5, count - mate_one_count - mate_two_count),
    ):
        bases = json.loads((ROOT / "benchmark" / "heldout" / filename).read_text())
        for index in range(amount):
            base = bases[index % len(bases)]
            transform = (seed + index // len(bases)) % 4
            positions.append(_transform_position(base["fen"], base["best_move"], transform, depth))
    random.Random(seed ^ 0xC0FFEE).shuffle(positions)
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
    correct_by_depth = {1: 0, 3: 0, 5: 0}
    total_by_depth = {1: 0, 3: 0, 5: 0}
    for index, (fen, expected, depth) in enumerate(suite):
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
        total_by_depth[depth] += 1
        correct_by_depth[depth] += int(is_correct)
        results.append({"index": index, "fen": fen, "prediction": prediction,
                        "correct": is_correct, "legal": is_legal, "depth": depth, "error": error_name})
    inference_seconds = time.perf_counter() - started
    digest = hashlib.sha256("\n".join(fen for fen, _, _ in suite).encode()).hexdigest()
    metrics = {
        "tactical_pass_at_1": correct / count,
        "mate_in_one_pass_at_1": correct_by_depth[1] / total_by_depth[1],
        "mate_in_two_pass_at_1": correct_by_depth[3] / total_by_depth[3],
        "mate_in_three_pass_at_1": correct_by_depth[5] / total_by_depth[5],
        "illegal_move_rate": illegal / count,
        "crash_rate": crashes / count,
        "positions": float(count),
        "mean_legal_moves": legal_move_total / count,
        "inference_seconds": inference_seconds,
    }
    print(json.dumps({"metrics": metrics, "suite_hash": digest, "episodes": results}, sort_keys=True))


if __name__ == "__main__":
    main()
