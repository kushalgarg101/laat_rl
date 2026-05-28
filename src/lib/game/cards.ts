import type { Card, CardId, Suit } from "./types";

export const SUITS: Suit[] = ["S", "H", "D", "C"];
export const SUIT_NAMES: Record<Suit, string> = {
  S: "Spades",
  H: "Hearts",
  D: "Diamonds",
  C: "Clubs"
};

export const SUIT_MARKS: Record<Suit, string> = {
  S: "♠\uFE0E",
  H: "♥\uFE0E",
  D: "♦\uFE0E",
  C: "♣\uFE0E"
};

export const RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

export const RANK_LABELS: Record<number, string> = {
  2: "2",
  3: "3",
  4: "4",
  5: "5",
  6: "6",
  7: "7",
  8: "8",
  9: "9",
  10: "10",
  11: "J",
  12: "Q",
  13: "K",
  14: "A"
};

export const STARTING_CARD_ID = cardId("S", 14);

export function cardId(suit: Suit, rank: number): CardId {
  return SUITS.indexOf(suit) * RANKS.length + RANKS.indexOf(rank);
}

export function getCard(id: CardId): Card {
  const suitIndex = Math.floor(id / RANKS.length);
  const rankIndex = id % RANKS.length;
  return {
    id,
    suit: SUITS[suitIndex],
    rank: RANKS[rankIndex]
  };
}

export function createDeck(): CardId[] {
  return Array.from({ length: 52 }, (_, id) => id);
}

export function formatCard(id: CardId): string {
  const card = getCard(id);
  return `${RANK_LABELS[card.rank]}${SUIT_MARKS[card.suit]}`;
}

export function sortCards(cards: CardId[]): CardId[] {
  return [...cards].sort((a, b) => {
    const ca = getCard(a);
    const cb = getCard(b);
    if (ca.suit !== cb.suit) {
      return SUITS.indexOf(ca.suit) - SUITS.indexOf(cb.suit);
    }
    return ca.rank - cb.rank;
  });
}
