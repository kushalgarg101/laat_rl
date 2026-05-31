export type Suit = "S" | "H" | "D" | "C";

export type CardId = number;

export interface Card {
  id: CardId;
  suit: Suit;
  rank: number;
}

export interface PlayerState {
  id: number;
  name: string;
  hand: CardId[];
}

export interface TrickEntry {
  playerId: number;
  cardId: CardId;
}

export type GameEventType =
  | "game_started"
  | "card_played"
  | "laat"
  | "cards_collected"
  | "trick_discarded"
  | "round_ended"
  | "cards_redealt"
  | "game_ended"
  | "safety_stop";

export interface GameEvent {
  id: number;
  round: number;
  turn: number;
  type: GameEventType;
  message: string;
}

export interface GameConfig {
  playerCount: number;
  maxRounds: number;
  handThreshold: number;
  startingCardId: CardId;
  seed?: number;
  maxActionsPerGame: number;
}

export interface GameState {
  config: GameConfig;
  players: PlayerState[];
  currentPlayer: number;
  currentTrick: TrickEntry[];
  leadSuit: Suit | null;
  roundDiscard: CardId[];
  suitFailures: boolean[][];
  round: number;
  turn: number;
  eventSeq: number;
  firstMoveCardId: CardId | null;
  gameOver: boolean;
  winnerIds: number[];
  loserIds: number[];
  events: GameEvent[];
  roundRankings: Record<number, number[][]>;
}

export interface PublicPlayerView {
  id: number;
  name: string;
  handCount: number;
  hand: CardId[] | null;
  suitFailures: boolean[];
}

export interface PublicGameView {
  currentPlayer: number;
  currentTrick: TrickEntry[];
  leadSuit: Suit | null;
  roundDiscardCount: number;
  round: number;
  turn: number;
  gameOver: boolean;
  winnerIds: number[];
  loserIds: number[];
  players: PublicPlayerView[];
  events: GameEvent[];
  roundRankings: Record<number, number[][]>;
}
