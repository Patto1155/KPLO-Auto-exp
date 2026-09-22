"""Mutable chess policy used by the protected tactics benchmark.

Research agents may replace this implementation. The policy receives only the FEN,
legal UCI moves, and a deterministic seed; it never receives the answer.
"""

from __future__ import annotations

import chess


def choose_move(fen: str, legal_moves: list[str], seed: int) -> str:
    """Play a legal mate in one when available, otherwise use the baseline move."""
    del seed
    board = chess.Board(fen)
    for move_text in legal_moves:
        move = chess.Move.from_uci(move_text)
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move_text
    return sorted(legal_moves)[0]
