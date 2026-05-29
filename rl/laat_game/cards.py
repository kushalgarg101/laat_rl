from __future__ import annotations

from dataclasses import dataclass

SUITS = ("S", "H", "D", "C")
SUIT_NAMES = {
    "S": "Spades",
    "H": "Hearts",
    "D": "Diamonds",
    "C": "Clubs",
}
RANKS = tuple(range(2, 15))
STARTING_CARD_ID = 12


@dataclass(frozen=True)
class Card:
    id: int
    suit: str
    rank: int


def card_id(suit: str, rank: int) -> int:
    return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)


def get_card(card_id_: int) -> Card:
    suit_index = card_id_ // len(RANKS)
    rank_index = card_id_ % len(RANKS)
    return Card(card_id_, SUITS[suit_index], RANKS[rank_index])


def create_deck() -> list[int]:
    return list(range(52))


def sort_cards(cards: list[int]) -> list[int]:
    return sorted(cards, key=lambda card: (SUITS.index(get_card(card).suit), get_card(card).rank))


def format_card(card_id_: int) -> str:
    card = get_card(card_id_)
    rank = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(card.rank, str(card.rank))
    return f"{rank}{card.suit}"
