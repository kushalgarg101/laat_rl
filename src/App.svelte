<script lang="ts">
  import { onMount } from "svelte";
  import { formatCard, getCard, RANK_LABELS, sortCards, SUIT_NAMES, SUIT_MARKS, SUITS } from "./lib/game/cards";
  import { applyMoveNoResolve, finalizeTrick, createGame, getLegalMoves, getPublicView, runSafetyChecks } from "./lib/game/engine";
  import { chooseBotMove } from "./lib/bots/simpleBot";
  import type { CardId, GameState } from "./lib/game/types";

  type ApiView = {
    currentPlayer: number;
    currentTrick: { playerId: number; cardId: number }[];
    leadSuit: "S" | "H" | "D" | "C" | null;
    roundDiscardCount: number;
    round: number;
    turn: number;
    gameOver: boolean;
    winnerIds: number[];
    loserIds: number[];
    players: {
      id: number;
      name: string;
      handCount: number;
      hand: number[] | null;
      suitFailures: boolean[];
    }[];
    events: { round: number; turn: number; message: string }[];
    safetyErrors?: string[];
  };

  let playerCount = 4;
  let maxRounds = 12;
  let handThreshold = 52;
  let debugReveal = false;
  let rlMode = false;
  let rlRunning = false;
  let rlSessionId = "";
  let rlMessage = "";
  let rlLastAction: { actor: string; card: string } | null = null;
  let rlView: ApiView | null = null;
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
  $: displayView = rlMode && rlView ? rlView : view;
  $: humanHand = displayView.players[0]?.hand ?? [];
  $: legalMoves = rlMode ? [] : getLegalMoves(game, 0);
  $: safetyErrors = rlMode ? (rlView?.safetyErrors ?? []) : runSafetyChecks(game);

  function newGame() {
    errorMessage = "";
    isProcessing = false;
    rlMode = false;
    rlSessionId = "";
    rlView = null;
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

  async function startRlGame() {
    errorMessage = "";
    rlMessage = "Starting local RL agent session...";
    rlRunning = false;
    try {
      const response = await fetch("http://127.0.0.1:8001/api/agent-game/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_count: playerCount,
          max_rounds: maxRounds,
          hand_threshold: handThreshold,
          reveal_hands: debugReveal,
          model_path: "models/maskable_ppo_laat/latest.zip",
          device: "cuda"
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Failed to start RL session.");
      rlSessionId = data.sessionId;
      rlView = data.view;
      rlLastAction = data.lastAction;
      rlMode = true;
      rlMessage = `RL model loaded on ${data.device}.`;
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not connect to RL server.";
      rlMessage = "Start the Python RL server first.";
    }
  }

  async function stepRlGame() {
    if (!rlSessionId || rlRunning) return;
    rlRunning = true;
    errorMessage = "";
    try {
      const response = await fetch("http://127.0.0.1:8001/api/agent-game/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: rlSessionId, reveal_hands: debugReveal })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Failed to step RL session.");
      rlView = data.view;
      rlLastAction = data.lastAction;
      rlMessage = data.lastAction ? `${data.lastAction.actor} played ${data.lastAction.card}.` : "Game over.";
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not step RL session.";
    } finally {
      rlRunning = false;
    }
  }

  async function autoplayRlGame() {
    if (rlRunning) return;
    rlRunning = true;
    try {
      while (rlSessionId && rlView && !rlView.gameOver) {
        const response = await fetch("http://127.0.0.1:8001/api/agent-game/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: rlSessionId, reveal_hands: debugReveal })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail ?? "Failed to step RL session.");
        rlView = data.view;
        rlLastAction = data.lastAction;
        rlMessage = data.lastAction ? `${data.lastAction.actor} played ${data.lastAction.card}.` : "Game over.";
        await new Promise((resolve) => setTimeout(resolve, 650));
      }
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not autoplay RL session.";
    } finally {
      rlRunning = false;
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
      <button class="primary secondary" on:click={startRlGame}>RL demo</button>
      {#if rlMode}
        <button class="primary secondary" on:click={stepRlGame} disabled={rlRunning || displayView.gameOver}>Step</button>
        <button class="primary secondary" on:click={autoplayRlGame} disabled={rlRunning || displayView.gameOver}>Auto</button>
      {/if}
    </div>
  </section>

  {#if rlMode}
    <section class="agent-banner">
      <strong>RL Agent Mode</strong>
      <span>{rlMessage}</span>
      {#if rlLastAction}
        <span>Last action: {rlLastAction.actor} -> {rlLastAction.card}</span>
      {/if}
    </section>
  {/if}

  <div class="game-container">
    <div class="casino-table">
      <div class="felt">
        {#each displayView.players as player}
          <div class="player-avatar pos-{player.id}" class:active={player.id === displayView.currentPlayer}>
            {#if player.id !== 0}
              <img src="/assets/bot_avatar_{player.id}.png" alt="Avatar for {player.name}" class="avatar-image" />
            {/if}
            <div class="avatar-info">
              <strong>{player.name}</strong>
              <span>{player.handCount} cards</span>
              <p>{suitFailureText(player.suitFailures)}</p>
            </div>
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
          {#if displayView.currentTrick.length === 0}
            <p class="empty-trick">Lead any legal card to start.</p>
          {:else}
            {#each displayView.currentTrick as entry}
              <div class="played-card">
                <span>{displayView.players[entry.playerId].name}</span>
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
          <strong>{displayView.round}</strong>
        </div>
        <div class="status-box">
          <span>Turn</span>
          <strong>{displayView.turn}</strong>
        </div>
        <div class="status-box">
          <span>Lead</span>
          <strong>{displayView.leadSuit ? SUIT_NAMES[displayView.leadSuit] : "Open"}</strong>
        </div>
        <div class="status-box">
          <span>Discard</span>
          <strong>{displayView.roundDiscardCount}</strong>
        </div>
        <div class="status-box" style="grid-column: span 2;">
          <span>Current Player</span>
          <strong>{displayView.players[displayView.currentPlayer]?.name}</strong>
        </div>
      </div>

      <div class="log-panel">
        <h2>Event log</h2>
        <div class="events">
          {#each [...displayView.events].reverse() as event}
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
      {#each sortCards(humanHand) as cardId, index}
        {@const total = humanHand.length}
        {@const mid = (total - 1) / 2}
        {@const angle = (index - mid) * 8}
        {@const translateY = Math.abs(index - mid) * 3}
        {@const translateX = (index - mid) * 24}
        <button
          class={cardClass(cardId)}
          style="--card-rot: {angle}deg; --card-ty: {translateY}px; --card-tx: {translateX}px; z-index: {index};"
          class:disabled={!legalMoves.includes(cardId) || displayView.gameOver || isProcessing || rlMode}
          disabled={!legalMoves.includes(cardId) || displayView.gameOver || isProcessing || rlMode}
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
