import { getCard, SUITS } from "../game/cards";
import { applyMove, getLegalMoves } from "../game/engine";
import type { CardId, GameState, Suit } from "../game/types";

export function chooseBotMove(state: GameState): CardId {
  const legalMoves = getLegalMoves(state);
  if (legalMoves.length === 0) {
    throw new Error("Bot has no legal card move.");
  }
  if (state.leadSuit !== null) {
    return lowestRank(legalMoves);
  }

  const safeMoves = legalMoves.filter((cardId) => !suitHasKnownFailure(state, getCard(cardId).suit, state.currentPlayer));
  if (safeMoves.length > 0) {
    return lowestRank(safeMoves);
  }
  return lowestRank(legalMoves);
}

export function advanceBotsUntilHumanTurn(source: GameState, humanPlayerId = 0): GameState {
  let state = source;
  let guard = 0;

  while (!state.gameOver && state.currentPlayer !== humanPlayerId && guard < state.config.maxActionsPerGame) {
    state = applyMove(state, chooseBotMove(state));
    guard += 1;
  }

  return state;
}

function lowestRank(cards: CardId[]): CardId {
  return [...cards].sort((a, b) => getCard(a).rank - getCard(b).rank)[0];
}

function suitHasKnownFailure(state: GameState, suit: Suit, botId: number): boolean {
  const suitIndex = SUITS.indexOf(suit);
  return state.suitFailures.some((failures, playerId) => playerId !== botId && failures[suitIndex]);
}
