from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from rl.laat_game.cards import STARTING_CARD_ID, SUITS, SUIT_NAMES, create_deck, format_card, get_card, sort_cards


@dataclass
class GameConfig:
    player_count: int = 4
    max_rounds: int = 12
    hand_threshold: int = 52
    starting_card_id: int = STARTING_CARD_ID
    seed: int | None = None
    max_actions_per_game: int = 2000


@dataclass
class PlayerState:
    id: int
    name: str
    hand: list[int] = field(default_factory=list)


@dataclass
class TrickEntry:
    player_id: int
    card_id: int


@dataclass
class GameEvent:
    type: str
    message: str
    round: int
    turn: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    config: GameConfig
    players: list[PlayerState]
    current_player: int
    current_trick: list[TrickEntry] = field(default_factory=list)
    lead_suit: str | None = None
    round_discard: list[int] = field(default_factory=list)
    suit_failures: list[list[bool]] = field(default_factory=list)
    round: int = 1
    turn: int = 0
    first_move_card_id: int | None = STARTING_CARD_ID
    game_over: bool = False
    winner_ids: list[int] = field(default_factory=list)
    loser_ids: list[int] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)


def create_game(config: GameConfig | None = None) -> GameState:
    config = config or GameConfig()
    config.player_count = max(2, min(6, int(config.player_count)))
    rng = random.Random(config.seed)
    deck = create_deck()
    rng.shuffle(deck)

    players = [PlayerState(i, "You" if i == 0 else f"Bot {i}") for i in range(config.player_count)]
    deal_into_hands([player.hand for player in players], deck)
    for player in players:
        player.hand = sort_cards(player.hand)

    starting_player = next(
        (player.id for player in players if config.starting_card_id in player.hand),
        0,
    )
    state = GameState(
        config=config,
        players=players,
        current_player=starting_player,
        suit_failures=[[False for _ in SUITS] for _ in players],
        first_move_card_id=config.starting_card_id,
    )
    add_event(state, "game_started", f"{players[starting_player].name} starts with {format_card(config.starting_card_id)}.")
    return state


def get_legal_moves(state: GameState, player_id: int | None = None) -> list[int]:
    player_id = state.current_player if player_id is None else player_id
    if state.game_over or player_id != state.current_player:
        return []
    player = state.players[player_id]
    if not player.hand:
        return []
    if state.first_move_card_id is not None:
        return [state.first_move_card_id] if state.first_move_card_id in player.hand else []
    if state.lead_suit is None:
        return sort_cards(player.hand)
    suited = [card for card in player.hand if get_card(card).suit == state.lead_suit]
    return sort_cards(suited if suited else player.hand)


def apply_move(state: GameState, card_id: int) -> GameState:
    legal = get_legal_moves(state)
    if card_id not in legal:
        raise ValueError(f"{format_card(card_id)} is not legal for {state.players[state.current_player].name}.")

    player = state.players[state.current_player]
    player.hand.remove(card_id)
    played = get_card(card_id)
    lead_suit_before_play = state.lead_suit
    if state.lead_suit is None:
        state.lead_suit = played.suit

    state.current_trick.append(TrickEntry(player.id, card_id))
    state.first_move_card_id = None
    state.turn += 1
    is_laat_card = lead_suit_before_play is not None and played.suit != lead_suit_before_play
    if is_laat_card:
        add_event(state, "card_played", f"{player.name} gave {format_card(card_id)} as laat on {SUIT_NAMES[lead_suit_before_play]}.")
        collect_laat_trick(state)
    else:
        add_event(state, "card_played", f"{player.name} played {format_card(card_id)}.")
        if is_clean_trick_complete(state):
            discard_clean_trick(state)
        else:
            state.current_player = next_active_player(state, state.current_player)

    finish_round_if_needed(state)
    finish_game_if_needed(state)
    return state


def run_safety_checks(state: GameState) -> list[str]:
    all_cards = [card for player in state.players for card in player.hand]
    all_cards += state.round_discard
    all_cards += [entry.card_id for entry in state.current_trick]
    errors: list[str] = []
    if len(all_cards) != 52:
        errors.append(f"Expected 52 cards in state, found {len(all_cards)}.")
    if len(set(all_cards)) != 52:
        errors.append(f"Expected 52 unique cards, found {len(set(all_cards))}.")
    return errors


def snapshot(state: GameState) -> dict[str, Any]:
    return {
        "current_player": state.current_player,
        "lead_suit": state.lead_suit,
        "round": state.round,
        "turn": state.turn,
        "hands": [sort_cards(player.hand) for player in state.players],
        "trick": [(entry.player_id, entry.card_id) for entry in state.current_trick],
        "discard_count": len(state.round_discard),
        "suit_failures": state.suit_failures,
        "game_over": state.game_over,
        "winner_ids": state.winner_ids,
        "loser_ids": state.loser_ids,
    }


def choose_bot_move(state: GameState) -> int:
    legal = get_legal_moves(state)
    if not legal:
        raise ValueError("Bot has no legal move.")
    if state.lead_suit is not None:
        return lowest_rank(legal)
    safe_moves = [card for card in legal if not suit_has_known_failure(state, get_card(card).suit, state.current_player)]
    return lowest_rank(safe_moves if safe_moves else legal)


def advance_bots_until_agent_turn(state: GameState, agent_player_id: int = 0) -> GameState:
    guard = 0
    while not state.game_over and state.current_player != agent_player_id and guard < state.config.max_actions_per_game:
        apply_move(state, choose_bot_move(state))
        guard += 1
    return state


def lowest_rank(cards: list[int]) -> int:
    return min(cards, key=lambda card: get_card(card).rank)


def suit_has_known_failure(state: GameState, suit: str, player_id: int) -> bool:
    suit_index = SUITS.index(suit)
    return any(idx != player_id and failures[suit_index] for idx, failures in enumerate(state.suit_failures))


def discard_clean_trick(state: GameState) -> None:
    winner_id = current_highest_entry(state).player_id
    discarded = [entry.card_id for entry in state.current_trick]
    state.round_discard.extend(discarded)
    state.current_trick.clear()
    state.lead_suit = None
    state.current_player = winner_id if state.players[winner_id].hand else next_active_player(state, winner_id)
    add_event(
        state,
        "trick_discarded",
        f"{state.players[winner_id].name} wins the sub-round; {len(discarded)} card(s) go to the round discard.",
        {"winner_id": winner_id, "card_count": len(discarded)},
    )


def collect_laat_trick(state: GameState) -> None:
    failed_suit = state.lead_suit
    if failed_suit is None:
        return
    laat_entry = next((entry for entry in state.current_trick if get_card(entry.card_id).suit != failed_suit), None)
    if laat_entry is None:
        return

    failed_player = state.players[laat_entry.player_id]
    collector_id = current_highest_entry_for_suit(state, failed_suit).player_id
    collector = state.players[collector_id]
    collected_cards = [entry.card_id for entry in state.current_trick]

    state.suit_failures[failed_player.id][SUITS.index(failed_suit)] = True
    collector.hand = sort_cards(collector.hand + collected_cards)
    state.current_trick.clear()
    state.lead_suit = None
    state.current_player = collector_id

    add_event(
        state,
        "laat",
        f"{failed_player.name} gave {format_card(laat_entry.card_id)} as laat on {SUIT_NAMES[failed_suit]}; {collector.name} collects {len(collected_cards)} card(s).",
        {
            "failed_player_id": failed_player.id,
            "collector_id": collector_id,
            "card_count": len(collected_cards),
            "failed_suit": failed_suit,
            "laat_card_id": laat_entry.card_id,
        },
    )
    add_event(state, "cards_collected", f"{collector.name} restarts the sub-round.", {"collector_id": collector_id})


def is_clean_trick_complete(state: GameState) -> bool:
    active_ids = [
        player.id
        for player in state.players
        if player.hand or any(entry.player_id == player.id for entry in state.current_trick)
    ]
    played_ids = {entry.player_id for entry in state.current_trick}
    return bool(active_ids) and all(player_id in played_ids for player_id in active_ids)


def finish_round_if_needed(state: GameState) -> None:
    if state.game_over or state.current_trick:
        return
    active_count = sum(1 for player in state.players if player.hand)
    if active_count > 1 or not state.round_discard:
        return

    add_event(state, "round_ended", f"Round {state.round} ended with {active_count} player(s) holding cards.")
    if state.round >= state.config.max_rounds:
        finish_by_largest_hand(state)
        return

    rng = random.Random((state.config.seed or 0) + state.round * 9973 + state.turn)
    redeal_cards = list(state.round_discard)
    rng.shuffle(redeal_cards)
    state.round_discard.clear()
    deal_into_hands([player.hand for player in state.players], redeal_cards)
    for player in state.players:
        player.hand = sort_cards(player.hand)

    state.round += 1
    state.suit_failures = [[False for _ in SUITS] for _ in state.players]
    state.current_player = choose_round_leader(state)
    add_event(state, "cards_redealt", f"Round {state.round} starts after redealing {len(redeal_cards)} discarded card(s).")


def finish_game_if_needed(state: GameState) -> None:
    if state.game_over:
        return
    if state.turn >= state.config.max_actions_per_game:
        add_event(state, "safety_stop", "Game stopped by safety limit.")
        finish_by_largest_hand(state)
        return
    if any(len(player.hand) >= state.config.hand_threshold for player in state.players):
        finish_by_largest_hand(state)


def finish_by_largest_hand(state: GameState) -> None:
    counts = [len(player.hand) for player in state.players]
    max_cards = max(counts)
    min_cards = min(counts)
    state.loser_ids = [player.id for player in state.players if len(player.hand) == max_cards]
    state.winner_ids = [player.id for player in state.players if len(player.hand) == min_cards]
    state.game_over = True
    add_event(state, "game_ended", f"Game ended. Most cards: {max_cards}. Fewest cards: {min_cards}.")


def current_highest_entry(state: GameState) -> TrickEntry:
    if not state.current_trick:
        raise ValueError("Cannot find highest card without a current trick.")
    return max(state.current_trick, key=lambda entry: get_card(entry.card_id).rank)


def current_highest_entry_for_suit(state: GameState, suit: str) -> TrickEntry:
    suited_entries = [entry for entry in state.current_trick if get_card(entry.card_id).suit == suit]
    if not suited_entries:
        raise ValueError(f"Cannot find highest {SUIT_NAMES[suit]} card without a suited trick card.")
    return max(suited_entries, key=lambda entry: get_card(entry.card_id).rank)


def next_active_player(state: GameState, from_player: int) -> int:
    for offset in range(1, len(state.players) + 1):
        candidate = (from_player + offset) % len(state.players)
        if state.players[candidate].hand:
            return candidate
    return from_player


def choose_round_leader(state: GameState) -> int:
    return next((player.id for player in state.players if player.hand), 0)


def deal_into_hands(hands: list[list[int]], cards: list[int]) -> None:
    for index, card in enumerate(cards):
        hands[index % len(hands)].append(card)


def add_event(state: GameState, type_: str, message: str, metadata: dict[str, Any] | None = None) -> None:
    state.events.append(GameEvent(type_, message, state.round, state.turn, metadata or {}))
