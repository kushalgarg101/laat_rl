import { createDeck, formatCard, getCard, sortCards, STARTING_CARD_ID, SUIT_NAMES, SUITS } from "./cards";
import { createRng, shuffle } from "./random";
import type { CardId, GameConfig, GameEventType, GameState, PublicGameView, Suit } from "./types";

export function defaultConfig(overrides: Partial<GameConfig> = {}): GameConfig {
  return {
    playerCount: 4,
    maxRounds: 12,
    handThreshold: 52,
    startingCardId: STARTING_CARD_ID,
    maxActionsPerGame: 2000,
    ...overrides
  };
}

export function createGame(configOverrides: Partial<GameConfig> = {}): GameState {
  const config = defaultConfig(configOverrides);
  const playerCount = clamp(Math.floor(config.playerCount), 2, 6);
  config.playerCount = playerCount;

  const rng = createRng(config.seed);
  const deck = shuffle(createDeck(), rng);
  const players = Array.from({ length: playerCount }, (_, id) => ({
    id,
    name: id === 0 ? "You" : `Bot ${id}`,
    hand: [] as CardId[]
  }));

  dealIntoHands(players.map((player) => player.hand), deck);
  for (const player of players) {
    player.hand = sortCards(player.hand);
  }

  const startingPlayer = players.find((player) => player.hand.includes(config.startingCardId))?.id ?? 0;
  const state: GameState = {
    config,
    players,
    currentPlayer: startingPlayer,
    currentTrick: [],
    leadSuit: null,
    roundDiscard: [],
    suitFailures: Array.from({ length: playerCount }, () => SUITS.map(() => false)),
    round: 1,
    turn: 0,
    eventSeq: 0,
    firstMoveCardId: config.startingCardId,
    gameOver: false,
    winnerIds: [],
    loserIds: [],
    events: [],
    roundRankings: {}
  };

  addEvent(state, "game_started", `${state.players[startingPlayer].name} starts with ${formatCard(config.startingCardId)}.`);
  return state;
}

export function cloneGame(state: GameState): GameState {
  return {
    ...state,
    config: { ...state.config },
    players: state.players.map((player) => ({ ...player, hand: [...player.hand] })),
    currentTrick: state.currentTrick.map((entry) => ({ ...entry })),
    roundDiscard: [...state.roundDiscard],
    suitFailures: state.suitFailures.map((row) => [...row]),
    winnerIds: [...state.winnerIds],
    loserIds: [...state.loserIds],
    events: state.events.map((event) => ({ ...event })),
    roundRankings: { ...state.roundRankings }
  };
}

export function getLegalMoves(state: GameState, playerId = state.currentPlayer): CardId[] {
  if (state.gameOver || playerId !== state.currentPlayer) {
    return [];
  }
  const player = state.players[playerId];
  if (!player || player.hand.length === 0) {
    return [];
  }
  if (state.firstMoveCardId !== null) {
    return player.hand.includes(state.firstMoveCardId) ? [state.firstMoveCardId] : [];
  }
  if (state.leadSuit === null) {
    return sortCards(player.hand);
  }
  const suited = player.hand.filter((cardId) => getCard(cardId).suit === state.leadSuit);
  return suited.length > 0 ? sortCards(suited) : sortCards(player.hand);
}

export function playerMustGiveLaat(state: GameState): boolean {
  return false;
}

export function applyMove(source: GameState, cardId: CardId): GameState {
  const state = cloneGame(source);
  if (state.gameOver) {
    return state;
  }

  const legalMoves = getLegalMoves(state);
  if (!legalMoves.includes(cardId)) {
    throw new Error(`${formatCard(cardId)} is not legal for ${state.players[state.currentPlayer].name}.`);
  }

  const player = state.players[state.currentPlayer];
  player.hand = player.hand.filter((heldCard) => heldCard !== cardId);
  const played = getCard(cardId);
  const leadSuitBeforePlay = state.leadSuit;
  if (state.leadSuit === null) {
    state.leadSuit = played.suit;
  }
  state.currentTrick.push({ playerId: player.id, cardId });
  state.firstMoveCardId = null;
  state.turn += 1;
  const isLaatCard = leadSuitBeforePlay !== null && played.suit !== leadSuitBeforePlay;
  addEvent(
    state,
    "card_played",
    isLaatCard
      ? `${player.name} gave ${formatCard(cardId)} as laat on ${SUIT_NAMES[leadSuitBeforePlay]}.`
      : `${player.name} played ${formatCard(cardId)}.`
  );

  if (isLaatCard) {
    collectLaatTrick(state);
  } else if (isCleanTrickComplete(state)) {
    discardCleanTrick(state);
  } else {
    state.currentPlayer = nextActivePlayer(state, state.currentPlayer);
  }

  finishRoundIfNeeded(state);
  finishGameIfNeeded(state);
  return state;
}

// Play a card and add it to the trick, but do NOT resolve/clear the trick yet.
// Returns { state, trickComplete } so the UI can pause before finalizing.
export function applyMoveNoResolve(source: GameState, cardId: CardId): { state: GameState; trickComplete: boolean } {
  const state = cloneGame(source);
  if (state.gameOver) {
    return { state, trickComplete: false };
  }

  const legalMoves = getLegalMoves(state);
  if (!legalMoves.includes(cardId)) {
    throw new Error(`${formatCard(cardId)} is not legal for ${state.players[state.currentPlayer].name}.`);
  }

  const player = state.players[state.currentPlayer];
  player.hand = player.hand.filter((heldCard) => heldCard !== cardId);
  const played = getCard(cardId);
  const leadSuitBeforePlay = state.leadSuit;
  if (state.leadSuit === null) {
    state.leadSuit = played.suit;
  }
  state.currentTrick.push({ playerId: player.id, cardId });
  state.firstMoveCardId = null;
  state.turn += 1;
  const isLaatCard = leadSuitBeforePlay !== null && played.suit !== leadSuitBeforePlay;
  addEvent(
    state,
    "card_played",
    isLaatCard
      ? `${player.name} gave ${formatCard(cardId)} as laat on ${SUIT_NAMES[leadSuitBeforePlay]}.`
      : `${player.name} played ${formatCard(cardId)}.`
  );

  const trickComplete = isLaatCard || isCleanTrickComplete(state);
  if (!trickComplete) {
    state.currentPlayer = nextActivePlayer(state, state.currentPlayer);
  }
  // If trickComplete, don't touch currentPlayer yet — caller will call finalizeTrick.

  return { state, trickComplete };
}

// Resolve and clear a completed trick, then finish round/game if needed.
export function finalizeTrick(source: GameState): GameState {
  const state = cloneGame(source);
  if (hasLaatCard(state)) {
    collectLaatTrick(state);
  } else if (isCleanTrickComplete(state)) {
    discardCleanTrick(state);
  }
  finishRoundIfNeeded(state);
  finishGameIfNeeded(state);
  return state;
}

export function resolveLaat(source: GameState): GameState {
  return finalizeTrick(source);
}

export function getPublicView(state: GameState, viewerId: number, debugReveal = false): PublicGameView {
  return {
    currentPlayer: state.currentPlayer,
    currentTrick: state.currentTrick.map((entry) => ({ ...entry })),
    leadSuit: state.leadSuit,
    roundDiscardCount: state.roundDiscard.length,
    round: state.round,
    turn: state.turn,
    gameOver: state.gameOver,
    winnerIds: [...state.winnerIds],
    loserIds: [...state.loserIds],
    players: state.players.map((player) => ({
      id: player.id,
      name: player.name,
      handCount: player.hand.length,
      hand: player.id === viewerId || debugReveal ? sortCards(player.hand) : null,
      suitFailures: [...state.suitFailures[player.id]]
    })),
    events: state.events.slice(-80),
    roundRankings: state.roundRankings
  };
}

export function runSafetyChecks(state: GameState): string[] {
  const errors: string[] = [];
  const allCards = state.players.flatMap((player) => player.hand).concat(state.roundDiscard, state.currentTrick.map((entry) => entry.cardId));
  const unique = new Set(allCards);
  if (allCards.length !== 52) {
    errors.push(`Expected 52 cards in state, found ${allCards.length}.`);
  }
  if (unique.size !== 52) {
    errors.push(`Expected 52 unique cards, found ${unique.size}.`);
  }
  return errors;
}

function discardCleanTrick(state: GameState): void {
  const winnerId = currentHighestEntry(state).playerId;
  const discarded = state.currentTrick.map((entry) => entry.cardId);
  state.roundDiscard = [...state.roundDiscard, ...discarded];
  state.currentTrick = [];
  state.leadSuit = null;
  state.currentPlayer = state.players[winnerId].hand.length > 0 ? winnerId : nextActivePlayer(state, winnerId);
  addEvent(state, "trick_discarded", `${state.players[winnerId].name} wins the sub-round; ${discarded.length} card(s) go to the round discard.`);
}

function collectLaatTrick(state: GameState): void {
  const failedSuit = state.leadSuit as Suit;
  const laatEntry = state.currentTrick.find((entry) => getCard(entry.cardId).suit !== failedSuit);
  if (!laatEntry) {
    return;
  }

  const failedPlayer = state.players[laatEntry.playerId];
  const collectorId = currentHighestEntryForSuit(state, failedSuit).playerId;
  const collector = state.players[collectorId];
  const collectedCards = state.currentTrick.map((entry) => entry.cardId);

  state.suitFailures[failedPlayer.id][SUITS.indexOf(failedSuit)] = true;
  collector.hand = sortCards([...collector.hand, ...collectedCards]);
  state.currentTrick = [];
  state.leadSuit = null;
  state.currentPlayer = collectorId;

  addEvent(
    state,
    "laat",
    `${failedPlayer.name} gave ${formatCard(laatEntry.cardId)} as laat on ${SUIT_NAMES[failedSuit]}; ${collector.name} collects ${collectedCards.length} card(s).`
  );
  addEvent(state, "cards_collected", `${collector.name} restarts the sub-round.`);
}

function hasLaatCard(state: GameState): boolean {
  if (state.leadSuit === null) {
    return false;
  }
  return state.currentTrick.some((entry) => getCard(entry.cardId).suit !== state.leadSuit);
}

function isCleanTrickComplete(state: GameState): boolean {
  const activeIds = state.players.filter((player) => player.hand.length > 0 || state.currentTrick.some((entry) => entry.playerId === player.id)).map((player) => player.id);
  const playedIds = new Set(state.currentTrick.map((entry) => entry.playerId));
  return activeIds.length > 0 && activeIds.every((id) => playedIds.has(id));
}

function finishRoundIfNeeded(state: GameState): void {
  if (state.gameOver || state.currentTrick.length > 0) {
    return;
  }
  const activeCount = state.players.filter((player) => player.hand.length > 0).length;
  if (activeCount > 1 || state.roundDiscard.length === 0) {
    return;
  }

  // Calculate rankings for this round (fewest cards = 1st place)
  const counts = new Map<number, number[]>();
  for (const player of state.players) {
    const c = player.hand.length;
    if (!counts.has(c)) {
      counts.set(c, []);
    }
    counts.get(c)!.push(player.id);
  }
  const rankedGroups = Array.from(counts.keys())
    .sort((a, b) => a - b)
    .map((c) => counts.get(c)!);
  state.roundRankings[state.round] = rankedGroups;

  addEvent(state, "round_ended", `Round ${state.round} ended with ${activeCount} player(s) holding cards.`);

  if (state.round >= state.config.maxRounds) {
    finishByLargestHand(state);
    return;
  }

  const rng = createRng((state.config.seed ?? Date.now()) + state.round * 9973 + state.turn);
  const redealCards = shuffle(state.roundDiscard, rng);
  state.roundDiscard = [];
  dealIntoHands(state.players.map((player) => player.hand), redealCards);
  for (const player of state.players) {
    player.hand = sortCards(player.hand);
  }

  state.round += 1;
  state.suitFailures = Array.from({ length: state.players.length }, () => SUITS.map(() => false));
  state.currentPlayer = chooseRoundLeader(state);
  addEvent(state, "cards_redealt", `Round ${state.round} starts after redealing ${redealCards.length} discarded card(s).`);
}

function finishGameIfNeeded(state: GameState): void {
  if (state.gameOver) {
    return;
  }
  if (state.turn >= state.config.maxActionsPerGame) {
    addEvent(state, "safety_stop", "Game stopped by safety limit.");
    finishByLargestHand(state);
    return;
  }
  if (state.players.some((player) => player.hand.length >= state.config.handThreshold)) {
    finishByLargestHand(state);
  }
}

function finishByLargestHand(state: GameState): void {
  const counts = state.players.map((player) => player.hand.length);
  const maxCards = Math.max(...counts);
  const minCards = Math.min(...counts);
  state.loserIds = state.players.filter((player) => player.hand.length === maxCards).map((player) => player.id);
  state.winnerIds = state.players.filter((player) => player.hand.length === minCards).map((player) => player.id);
  state.gameOver = true;
  addEvent(state, "game_ended", `Game ended. Most cards: ${maxCards}. Fewest cards: ${minCards}.`);
}

function currentHighestEntry(state: GameState) {
  if (state.currentTrick.length === 0) {
    throw new Error("Cannot find highest card without a current trick.");
  }
  return state.currentTrick.reduce((best, entry) => {
    const bestCard = getCard(best.cardId);
    const card = getCard(entry.cardId);
    return card.rank > bestCard.rank ? entry : best;
  });
}

function currentHighestEntryForSuit(state: GameState, suit: Suit) {
  const suitedEntries = state.currentTrick.filter((entry) => getCard(entry.cardId).suit === suit);
  if (suitedEntries.length === 0) {
    throw new Error(`Cannot find highest ${SUIT_NAMES[suit]} card without a suited trick card.`);
  }
  return suitedEntries.reduce((best, entry) => {
    const bestCard = getCard(best.cardId);
    const card = getCard(entry.cardId);
    return card.rank > bestCard.rank ? entry : best;
  });
}

function nextActivePlayer(state: GameState, fromPlayer: number): number {
  for (let offset = 1; offset <= state.players.length; offset += 1) {
    const candidate = (fromPlayer + offset) % state.players.length;
    if (state.players[candidate].hand.length > 0) {
      return candidate;
    }
  }
  return fromPlayer;
}

function chooseRoundLeader(state: GameState): number {
  const withCards = state.players.find((player) => player.hand.length > 0);
  return withCards?.id ?? 0;
}

function dealIntoHands(hands: CardId[][], cards: CardId[]): void {
  cards.forEach((cardId, index) => {
    hands[index % hands.length].push(cardId);
  });
}

function addEvent(state: GameState, type: GameEventType, message: string): void {
  state.eventSeq += 1;
  state.events.push({
    id: state.eventSeq,
    round: state.round,
    turn: state.turn,
    type,
    message
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
