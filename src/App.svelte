<script lang="ts">
  import { onMount } from "svelte";
  import { formatCard, getCard, RANK_LABELS, sortCards, SUIT_NAMES, SUIT_MARKS, SUITS } from "./lib/game/cards";
  import { applyMoveNoResolve, finalizeTrick, createGame, getLegalMoves, getPublicView, runSafetyChecks } from "./lib/game/engine";
  import { chooseBotMove } from "./lib/bots/simpleBot";
  import type { CardId, GameState } from "./lib/game/types";

  let playerCount = 4;
  let maxRounds = 12;
  let handThreshold = 52;
  let debugReveal = false;
  let errorMessage = "";
  let game: GameState = createGame({ playerCount, maxRounds, handThreshold });
  let isProcessing = false;

  const BOT_DELAY_MS = 700;   // pause between each bot card
  const TRICK_SHOW_MS = 1200; // extra pause to show all cards before clearing

  // Play one card (either human or bot), show it on the table,
  // and if it completes the trick pause before resolving.
  async function playOneCard(cardId: CardId) {
    const { state, trickComplete } = applyMoveNoResolve(game, cardId);
    game = state; // → UI now shows the card on the table

    if (trickComplete) {
      await new Promise(r => setTimeout(r, TRICK_SHOW_MS)); // hold all cards visible
      game = finalizeTrick(game);                           // clear trick, advance
    }
  }

  async function runBotTurns() {
    if (isProcessing) return;
    isProcessing = true;

    let guard = 0;
    while (!game.gameOver && game.currentPlayer !== 0 && guard < game.config.maxActionsPerGame) {
      await new Promise(r => setTimeout(r, BOT_DELAY_MS));

      await playOneCard(chooseBotMove(game));
      guard += 1;
    }

    isProcessing = false;
  }

  onMount(() => { runBotTurns(); });

  $: view = getPublicView(game, 0, debugReveal);
  $: humanHand = view.players[0]?.hand ?? [];
  $: legalMoves = getLegalMoves(game, 0);
  $: safetyErrors = runSafetyChecks(game);

  function newGame() {
    errorMessage = "";
    isProcessing = false;
    game = createGame({ playerCount, maxRounds, handThreshold });
    runBotTurns();
  }

  async function playCard(cardId: CardId) {
    if (isProcessing) return;
    errorMessage = "";
    try {
      await playOneCard(cardId); // shows human card, pauses if trick complete
      await runBotTurns();       // then bots respond
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Invalid move.";
      isProcessing = false;
    }
  }

  function cardClass(cardId: CardId) {
    const suit = getCard(cardId).suit;
    return `card ${suit === "H" || suit === "D" ? "red" : "black"}`;
  }

  function suitFailureText(failures: boolean[]) {
    const failed = failures.map((value, index) => (value ? SUIT_MARKS[SUITS[index]] : null)).filter(Boolean);
    return failed.length === 0 ? "No known suit failures" : `No ${failed.join(", ")}`;
  }
</script>

<main class="app-shell">
  <section class="topbar">
    <div>
      <p class="eyebrow">Laat simulator</p>
      <h1>Hidden-information card table</h1>
    </div>
    <div class="controls">
      <label>
        Players
        <input type="number" min="2" max="6" bind:value={playerCount} />
      </label>
      <label>
        Max rounds
        <input type="number" min="1" max="200" bind:value={maxRounds} />
      </label>
      <label>
        End threshold
        <input type="number" min="1" max="52" bind:value={handThreshold} />
      </label>
      <label class="toggle">
        <input type="checkbox" bind:checked={debugReveal} />
        Reveal hands
      </label>
      <button class="primary" on:click={newGame}>New game</button>
    </div>
  </section>

  <div class="game-container">
    <div class="casino-table">
      <div class="felt">
        {#each view.players as player}
          <div class="player-avatar pos-{player.id}" class:active={player.id === view.currentPlayer}>
            <strong>{player.name}</strong>
            <span>{player.handCount} cards</span>
            <p>{suitFailureText(player.suitFailures)}</p>
            {#if player.hand && player.id !== 0}
              <div class="mini-hand" style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: center; margin-top: 8px;">
                {#each sortCards(player.hand) as cardId}
                  <span class={cardClass(cardId)} style="background: white; padding: 2px 4px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; border: 1px solid #ccc;">{formatCard(cardId)}</span>
                {/each}
              </div>
            {:else if player.id !== 0 && player.handCount > 0}
              <div class="hidden-hand">
                {#each Array.from({length: player.handCount}) as _}
                  <div class="card-back"></div>
                {/each}
              </div>
            {/if}
          </div>
        {/each}

        <div class="trick">
          {#if view.currentTrick.length === 0}
            <p class="empty-trick">Lead any legal card to start.</p>
          {:else}
            {#each view.currentTrick as entry}
              <div class="played-card">
                <span>{view.players[entry.playerId].name}</span>
                <b class={cardClass(entry.cardId)}>
                  <div class="card-top">
                    <span class="rank">{RANK_LABELS[getCard(entry.cardId).rank]}</span>
                    <span class="suit">{SUIT_MARKS[getCard(entry.cardId).suit]}</span>
                  </div>
                  <div class="center-suit">{SUIT_MARKS[getCard(entry.cardId).suit]}</div>
                  <div class="card-bottom">
                    <span class="rank">{RANK_LABELS[getCard(entry.cardId).rank]}</span>
                    <span class="suit">{SUIT_MARKS[getCard(entry.cardId).suit]}</span>
                  </div>
                </b>
              </div>
            {/each}
          {/if}
        </div>
      </div>
    </div>

    <aside class="side-panel">
      <div class="status-grid">
        <div class="status-box">
          <span>Round</span>
          <strong>{view.round}</strong>
        </div>
        <div class="status-box">
          <span>Turn</span>
          <strong>{view.turn}</strong>
        </div>
        <div class="status-box">
          <span>Lead</span>
          <strong>{view.leadSuit ? SUIT_NAMES[view.leadSuit] : "Open"}</strong>
        </div>
        <div class="status-box">
          <span>Discard</span>
          <strong>{view.roundDiscardCount}</strong>
        </div>
        <div class="status-box" style="grid-column: span 2;">
          <span>Current Player</span>
          <strong>{view.players[view.currentPlayer]?.name}</strong>
        </div>
      </div>

      <div class="log-panel">
        <h2>Event log</h2>
        <div class="events">
          {#each [...view.events].reverse() as event}
            <p><span>R{event.round} T{event.turn}</span>{event.message}</p>
          {/each}
        </div>
      </div>
    </aside>
  </div>

  <section class="hand-section">
    <div class="hand-heading">
      <div>
        <p class="eyebrow">Your hand</p>
        <h2>{humanHand.length} cards</h2>
      </div>
      {#if isProcessing}
        <p class="thinking">Bots are playing…</p>
      {:else}
        <p>{legalMoves.length} legal move{legalMoves.length === 1 ? "" : "s"}</p>
      {/if}
    </div>
    <div class="hand">
      {#each sortCards(humanHand) as cardId}
        <button
          class={cardClass(cardId)}
          class:disabled={!legalMoves.includes(cardId) || view.gameOver || isProcessing}
          disabled={!legalMoves.includes(cardId) || view.gameOver || isProcessing}
          on:click={() => playCard(cardId)}
          title={`${RANK_LABELS[getCard(cardId).rank]} of ${SUIT_NAMES[getCard(cardId).suit]}`}
        >
          <div class="card-top">
            <span class="rank">{RANK_LABELS[getCard(cardId).rank]}</span>
            <span class="suit">{SUIT_MARKS[getCard(cardId).suit]}</span>
          </div>
          <div class="center-suit">{SUIT_MARKS[getCard(cardId).suit]}</div>
          <div class="card-bottom">
            <span class="rank">{RANK_LABELS[getCard(cardId).rank]}</span>
            <span class="suit">{SUIT_MARKS[getCard(cardId).suit]}</span>
          </div>
        </button>
      {/each}
    </div>
  </section>
</main>
