from __future__ import annotations

import numpy as np

from rl.laat_game.cards import SUITS, get_card
from rl.laat_game.engine import GameState

MAX_PLAYERS = 6
OBS_SIZE = 52 + (MAX_PLAYERS * 52) + MAX_PLAYERS + (MAX_PLAYERS * 4) + MAX_PLAYERS + 5 + MAX_PLAYERS + 3


def relative_player_id(player_id: int, perspective_player: int, player_count: int) -> int:
    return (player_id - perspective_player) % player_count


def encode_observation(state: GameState, perspective_player: int = 0) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    offset = 0
    player_count = len(state.players)

    for card in state.players[perspective_player].hand:
        obs[offset + card] = 1.0
    offset += 52

    for entry in state.current_trick:
        relative_id = relative_player_id(entry.player_id, perspective_player, player_count)
        if relative_id < MAX_PLAYERS:
            obs[offset + relative_id * 52 + entry.card_id] = 1.0
    offset += MAX_PLAYERS * 52

    for absolute_offset in range(min(player_count, MAX_PLAYERS)):
        player_id = (perspective_player + absolute_offset) % player_count
        obs[offset + absolute_offset] = len(state.players[player_id].hand) / 52.0
    offset += MAX_PLAYERS

    for absolute_offset in range(min(player_count, MAX_PLAYERS)):
        player_id = (perspective_player + absolute_offset) % player_count
        for suit_index, failed in enumerate(state.suit_failures[player_id]):
            obs[offset + absolute_offset * 4 + suit_index] = 1.0 if failed else 0.0
    offset += MAX_PLAYERS * 4

    relative_current = relative_player_id(state.current_player, perspective_player, player_count)
    if relative_current < MAX_PLAYERS:
        obs[offset + relative_current] = 1.0
    offset += MAX_PLAYERS

    if state.lead_suit is None:
        obs[offset + 4] = 1.0
    else:
        obs[offset + SUITS.index(state.lead_suit)] = 1.0
    offset += 5

    highest_player = current_highest_player(state)
    if highest_player is not None:
        relative_highest = relative_player_id(highest_player, perspective_player, player_count)
        if relative_highest < MAX_PLAYERS:
            obs[offset + relative_highest] = 1.0
    offset += MAX_PLAYERS

    obs[offset] = min(state.round / max(state.config.max_rounds, 1), 1.0)
    obs[offset + 1] = min(state.turn / max(state.config.max_actions_per_game, 1), 1.0)
    obs[offset + 2] = len(state.round_discard) / 52.0
    return obs


def action_mask(state: GameState, player_id: int) -> np.ndarray:
    from rl.laat_game.engine import get_legal_moves

    mask = np.zeros(52, dtype=bool)
    for action in get_legal_moves(state, player_id):
        mask[action] = True
    return mask


def current_highest_player(state: GameState) -> int | None:
    if state.lead_suit is None or not state.current_trick:
        return None
    suited_entries = [
        entry for entry in state.current_trick
        if get_card(entry.card_id).suit == state.lead_suit
    ]
    if not suited_entries:
        return None
    return max(suited_entries, key=lambda entry: get_card(entry.card_id).rank).player_id
