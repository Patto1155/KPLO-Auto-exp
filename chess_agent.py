"""Mutable chess policy used by the protected tactics benchmark.

Research agents may replace this implementation. The policy receives only the FEN,
legal UCI moves, and a deterministic seed; it never receives the answer.
"""

from __future__ import annotations

import chess

_FORCE_CACHE: dict[tuple[str, int], bool] = {}


def _forces_mate(board: chess.Board, plies: int) -> bool:
    key = (board.fen(), plies)
    if key in _FORCE_CACHE:
        return _FORCE_CACHE[key]
    if board.is_checkmate():
        return True
    if plies == 0 or board.is_stalemate():
        return False
    outcomes = []
    for reply in list(board.legal_moves):
        board.push(reply)
        outcomes.append(_forces_mate(board, plies - 1))
        board.pop()
    result = any(outcomes) if board.turn == chess.WHITE else bool(outcomes) and all(outcomes)
    _FORCE_CACHE[key] = result
    return result


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
    for move_text in legal_moves:
        move = chess.Move.from_uci(move_text)
        board.push(move)
        forces_mate = _forces_mate(board, 2)
        board.pop()
        if forces_mate:
            return move_text
    if any(piece.piece_type == chess.QUEEN for piece in board.piece_map().values()):
        for move_text in legal_moves:
            move = chess.Move.from_uci(move_text)
            board.push(move)
            forces_mate = _forces_mate(board, 4)
            board.pop()
            if forces_mate:
                return move_text
    return sorted(legal_moves)[0]
