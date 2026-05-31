<script lang="ts">
  import { onMount } from "svelte";
  import { formatCard, getCard, RANK_LABELS, sortCards, SUIT_NAMES, SUIT_MARKS, SUITS } from "./lib/game/cards";
  import { applyMoveNoResolve, finalizeTrick, createGame, getLegalMoves, getPublicView, runSafetyChecks } from "./lib/game/engine";
  import { chooseBotMove } from "./lib/bots/simpleBot";
  import type { CardId, GameState } from "./lib/game/types";

  // ---------------------------------------------------------------------------
  // Types
  // ---------------------------------------------------------------------------

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
    roundRankings?: Record<number, number[][]>;
    round_rankings?: Record<number, number[][]>;
  };

  type ModelInfo = {
    id: string;
    label: string;
    description: string;
    path: string;
    available: boolean;
  };

  // ---------------------------------------------------------------------------
  // Game config
  // ---------------------------------------------------------------------------

  let playerCount = 4;
  let maxRounds = 12;
  let handThreshold = 52;
  let debugReveal = false;

  // ---------------------------------------------------------------------------
  // RL state
  // ---------------------------------------------------------------------------

  let rlPanelOpen = false;
  let rlMode = false;
  let rlRunning = false;
  let rlAutoMode = false; // True when user has toggled Auto play on
  let rlSessionId = "";
  let rlMessage = "";
  let rlLastAction: { actor: string; card: string; playerId?: number } | null = null;
  let rlView: ApiView | null = null;
  let rlPlayerModelLabels: Record<string, string> = {};

  // Per-player model selection: index = player slot, value = model id or ""
  // "human" means Player 0 plays manually, "" means heuristic bot, else model id
  let playerModelSelections: string[] = ["human", "", "", ""];

  // Available models fetched from server
  let availableModels: ModelInfo[] = [];
  let modelsLoaded = false;

  // ---------------------------------------------------------------------------
  // Session stats (persist across games)
  // ---------------------------------------------------------------------------

  type PlayerStat = { ranks: Record<number, number>; name: string };

  let sessionStats: {
    roundsPlayed: number;
    players: PlayerStat[];
  } = {
    roundsPlayed: 0,
    players: Array.from({ length: 6 }, (_, i) => ({ ranks: {}, name: `Player ${i}` })),
  };

  let lastProcessedRound = 0;
  let lastSessionId = "local-" + Date.now();

  function trackStats(view: any, sessionId: string) {
    if (lastSessionId !== sessionId) {
      lastProcessedRound = 0;
      lastSessionId = sessionId;
    }
    view.players.forEach(p => {
      sessionStats.players[p.id].name = p.name;
    });

    const roundRankings = view.roundRankings || view.round_rankings;
    console.log("=== trackStats debug ===", {
      roundRankings,
      keys: Object.keys(roundRankings || {}),
      lastProcessedRound,
      roundsPlayed: sessionStats.roundsPlayed,
      viewRound: view.round,
      gameOver: view.gameOver
    });
    const newRounds = Object.keys(roundRankings || {})
      .map(Number)
      .filter(r => r > lastProcessedRound)
      .sort((a, b) => a - b);

    for (const r of newRounds) {
      const rankedGroups = roundRankings[r];
      let currentRank = 1;
      for (const group of rankedGroups) {
        for (const pid of group) {
          // Clone the player and ranks objects to guarantee Svelte reactivity triggers
          const player = { ...sessionStats.players[pid] };
          player.ranks = { ...player.ranks };
          if (!player.ranks[currentRank]) {
            player.ranks[currentRank] = 0;
          }
          player.ranks[currentRank]++;
          sessionStats.players[pid] = player;
        }
        currentRank += group.length; 
      }
      sessionStats.roundsPlayed += 1;
      lastProcessedRound = r;
    }
    
    // Check for game over unfinished round
    if (view.gameOver && lastProcessedRound < view.round) {
      const counts = new Map<number, number[]>();
      for (const p of view.players) {
        const c = p.handCount ?? p.hand?.length ?? 0;
        if (!counts.has(c)) counts.set(c, []);
        counts.get(c)!.push(p.id);
      }
      const rankedGroups = Array.from(counts.keys()).sort((a,b)=>a-b).map(c => counts.get(c)!);
      let currentRank = 1;
      for (const group of rankedGroups) {
        for (const pid of group) {
          const player = { ...sessionStats.players[pid] };
          player.ranks = { ...player.ranks };
          if (!player.ranks[currentRank]) {
            player.ranks[currentRank] = 0;
          }
          player.ranks[currentRank]++;
          sessionStats.players[pid] = player;
        }
        currentRank += group.length;
      }
      sessionStats.roundsPlayed += 1;
      lastProcessedRound = view.round;
    }

    sessionStats.players = [...sessionStats.players];
    sessionStats = { ...sessionStats }; 
  }

  function resetStats() {
    sessionStats = {
      roundsPlayed: 0,
      players: Array.from({ length: 6 }, (_, i) => ({ ranks: {}, name: `Player ${i}` })),
    };
  }

  // Reactive: active player stats (only players in current game)
  $: activePlayerStats = sessionStats.players.slice(0, playerCount);

  // ---------------------------------------------------------------------------
  // Local game state
  // ---------------------------------------------------------------------------

  let errorMessage = "";
  let game: GameState = createGame({ playerCount, maxRounds, handThreshold });
  let localSessionId = "local-" + Date.now();
  let isProcessing = false;

  const BOT_DELAY_MS = 700;
  const TRICK_SHOW_MS = 1200;

  // Model colour palette for badges
  const MODEL_COLORS: Record<string, string> = {
    model1: "model-badge-1",
    model2: "model-badge-2",
    model3: "model-badge-3",
    human: "model-badge-human",
    "": "model-badge-none",
  };

  // Short labels for dropdowns
  const MODEL_SHORT: Record<string, string> = {
    "": "Heuristic Bot",
    human: "Human (You)",
    model1: "Model 1 – PPO",
    model2: "Model 2 – BC+PPO",
    model3: "Model 3 – League",
  };

  // ---------------------------------------------------------------------------
  // Reactive helpers
  // ---------------------------------------------------------------------------

  $: view = getPublicView(game, 0, debugReveal);
  $: displayView = rlMode && rlView ? rlView : view;
  
  $: {
    if (displayView) {
      trackStats(displayView, rlMode ? (rlSessionId || "rl-init") : localSessionId);
    }
  }

  $: humanHand = displayView.players[0]?.hand ?? [];

  // In RL mode, player 0 might be a model (not human) → disable hand clicking
  $: player0IsModel = rlMode && (playerModelSelections[0] !== "" && playerModelSelections[0] !== "human");
  // Allow clicking cards when: local game (no rlMode) OR it's a human RL turn
  // In RL mode during human turn, use the server-provided hand as all legal (server validates legality)
  $: legalMoves = rlMode
    ? (isHumanTurnInRl ? (rlView?.players[0]?.hand ?? []) : [])
    : getLegalMoves(game, 0);
  $: safetyErrors = rlMode ? (rlView?.safetyErrors ?? []) : runSafetyChecks(game);

  // Ensure playerModelSelections is always length playerCount
  $: {
    while (playerModelSelections.length < playerCount) playerModelSelections.push("");
    if (playerModelSelections.length > playerCount) playerModelSelections = playerModelSelections.slice(0, playerCount);
  }

  // ---------------------------------------------------------------------------
  // Local game helpers
  // ---------------------------------------------------------------------------

  async function playOneCard(cardId: CardId) {
    const { state, trickComplete } = applyMoveNoResolve(game, cardId);
    game = state;
    if (trickComplete) {
      await new Promise(r => setTimeout(r, TRICK_SHOW_MS));
      game = finalizeTrick(game);
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

  onMount(() => {
    runBotTurns();
    fetchModels();
  });

  function newGame() {
    errorMessage = "";
    isProcessing = false;
    rlMode = false;
    rlAutoMode = false;
    rlSessionId = "";
    rlView = null;
    rlPlayerModelLabels = {};
    game = createGame({ playerCount, maxRounds, handThreshold });
    localSessionId = "local-" + Date.now();
    runBotTurns();
  }

  async function playCard(cardId: CardId) {
    if (isProcessing) return;

    if (rlMode) {
      // Only allow card clicks if it's actually the human's turn
      if (!isHumanTurnInRl) return;
      await stepRlGame(cardId);
      // After the human plays, resume auto-play only if auto mode is active
      if (rlAutoMode && !rlView?.gameOver) {
        await autoplayRlGame();
      }
      return;
    }

    errorMessage = "";
    try {
      await playOneCard(cardId);
      await runBotTurns();
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Invalid move.";
      isProcessing = false;
    }
  }

  // ---------------------------------------------------------------------------
  // RL helpers
  // ---------------------------------------------------------------------------

  async function fetchModels() {
    try {
      const res = await fetch("http://127.0.0.1:8001/api/models");
      if (res.ok) {
        const data = await res.json();
        availableModels = data.models ?? [];
        modelsLoaded = true;
      }
    } catch {
      // Server not running yet — that's fine
    }
  }

  function toggleRlPanel() {
    rlPanelOpen = !rlPanelOpen;
    if (rlPanelOpen && !modelsLoaded) fetchModels();
  }

  async function startRlGame() {
    errorMessage = "";
    rlMessage = "Loading models…";
    rlRunning = false;
    rlMode = false;
    rlAutoMode = false;

    // Build player_models for server: "human" and "" both mean null (human/heuristic = no model)
    const playerModels = Array.from({ length: playerCount }, (_, i) => {
      const sel = playerModelSelections[i];
      return (sel === "" || sel === "human") ? null : sel;
    });

    try {
      const response = await fetch("http://127.0.0.1:8001/api/agent-game/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_count: playerCount,
          max_rounds: maxRounds,
          hand_threshold: handThreshold,
          reveal_hands: debugReveal,
          player_models: playerModels,
          device: "cuda",
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        throw new Error(msg ?? "Failed to start RL session.");
      }
      rlSessionId = data.sessionId;
      rlView = data.view;
      rlLastAction = data.lastAction;
      rlPlayerModelLabels = data.playerModelLabels ?? {};
      rlMode = true;
      rlMessage = `Session started on ${data.device}.`;
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not connect to RL server.";
      rlMessage = "Start the Python RL server first: uvicorn rl.server.agent_api:app --port 8001";
    }
  }

  async function stepRlGame(humanCardId?: number | Event) {
    if (!rlSessionId || rlRunning) return;
    rlRunning = true;
    errorMessage = "";
    try {
      const payload: any = { session_id: rlSessionId, reveal_hands: debugReveal };
      if (typeof humanCardId === "number") {
        payload.cardId = humanCardId;
      }
      const response = await fetch("http://127.0.0.1:8001/api/agent-game/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        throw new Error(msg ?? "Failed to step RL session.");
      }
      rlView = data.view;
      rlLastAction = data.lastAction;
      rlPlayerModelLabels = data.playerModelLabels ?? rlPlayerModelLabels;
      rlMessage = data.lastAction
        ? `${data.lastAction.actor} played ${data.lastAction.card}.`
        : "Game over.";
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not step RL session.";
    } finally {
      rlRunning = false;
    }
  }

  async function autoplayRlGame() {
    if (rlRunning) return;
    rlAutoMode = true;
    rlRunning = true;
    try {
      while (rlAutoMode && rlSessionId && rlView && !rlView.gameOver) {
        // Stop and wait for human input when it's human's turn
        if (rlView.currentPlayer === 0 && playerModelSelections[0] === "human") {
          break; // Keep rlAutoMode = true; will resume after human plays
        }
        const response = await fetch("http://127.0.0.1:8001/api/agent-game/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: rlSessionId, reveal_hands: debugReveal }),
        });
        const data = await response.json();
        if (!response.ok) {
          const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
          throw new Error(msg ?? "Failed to step RL session.");
        }
        rlView = data.view;
        rlLastAction = data.lastAction;
        rlPlayerModelLabels = data.playerModelLabels ?? rlPlayerModelLabels;
        rlMessage = data.lastAction
          ? `${data.lastAction.actor} played ${data.lastAction.card}.`
          : "Game over.";
        await new Promise((resolve) => setTimeout(resolve, 650));
      }
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : "Could not autoplay RL session.";
      rlAutoMode = false;
    } finally {
      rlRunning = false;
    }
  }

  function stopAutoRl() {
    rlAutoMode = false;
    // rlRunning will clear on its own after the current tick
  }

  // ---------------------------------------------------------------------------
  // Card display helpers
  // ---------------------------------------------------------------------------

  function cardClass(cardId: CardId) {
    const suit = getCard(cardId).suit;
    return `card ${suit === "H" || suit === "D" ? "red" : "black"}`;
  }

  function suitFailureText(failures: boolean[]) {
    const failed = failures
      .map((value, index) => (value ? SUIT_MARKS[SUITS[index]] : null))
      .filter(Boolean);
    return failed.length === 0 ? "No known suit failures" : `No ${failed.join(", ")}`;
  }

  function modelBadgeClass(modelId: string) {
    return MODEL_COLORS[modelId] ?? "model-badge-none";
  }

  function playerLabel(index: number): string {
    if (index === 0) return "You (Player 0)";
    return `Bot Player ${index}`;
  }

  // Helper to determine if we should show the "thinking" message
  $: isRlAgentThinking = rlMode && displayView && !isHumanTurnInRl && rlRunning;
  $: isHumanTurnInRl = rlMode && displayView && displayView.currentPlayer === 0 && playerModelSelections[0] === "human" && !rlRunning && !displayView.gameOver;

</script>

<main class="app-shell">
  <!-- ====================================================================
       Top bar
       ==================================================================== -->
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
      <button
        class="primary secondary"
        class:rl-panel-active={rlPanelOpen}
        on:click={toggleRlPanel}
      >
        🤖 RL Models {rlPanelOpen ? "▲" : "▼"}
      </button>

    </div>
  </section>

  <!-- ====================================================================
       Collapsible RL Model Setup Panel
       ==================================================================== -->
  {#if rlPanelOpen}
    <section class="rl-setup-panel">
      <div class="rl-setup-header">
        <div>
          <p class="eyebrow">RL Agent Configuration</p>
          <h2>Assign a model to each player slot</h2>
        </div>
        <p class="rl-hint">
          Set a model for any player slot. Leave blank for the heuristic bot.<br>
          <b>Player 0</b>: choose <em>Human</em> to play yourself, or an RL model for full autoplay.
        </p>
      </div>

      <div class="player-model-grid">
        {#each Array.from({ length: playerCount }, (_, i) => i) as pid}
          <div class="player-model-row" class:player0-row={pid === 0}>
            <div class="player-model-info">
              <span class="player-slot-label">
                {pid === 0 ? "🧑 You (Player 0)" : `🤖 Player ${pid}`}
              </span>
              {#if rlMode}
                <span class="current-model-badge {modelBadgeClass(playerModelSelections[pid])}">
                  {rlPlayerModelLabels[String(pid)] ?? "Heuristic Bot"}
                </span>
              {/if}
            </div>
            <div class="model-select-wrap">
              <select
                class="model-select {modelBadgeClass(playerModelSelections[pid])}"
                bind:value={playerModelSelections[pid]}
              >
                <option value="">— Heuristic Bot —</option>
                {#if pid === 0}
                  <option value="human">🧑 Human (Self Play)</option>
                {/if}
                <option value="model1">Model 1 – Masked PPO (Win ~84%)</option>
                <option value="model2">Model 2 – BC + PPO (Win ~95%)</option>
                <option value="model3">Model 3 – League PPO (Win ~90%)</option>
              </select>
              {#if playerModelSelections[pid]}
                <span class="model-pill {modelBadgeClass(playerModelSelections[pid])}">
                  {MODEL_SHORT[playerModelSelections[pid]] ?? playerModelSelections[pid]}
                </span>
              {/if}
            </div>
          </div>
        {/each}
      </div>

      <!-- Model descriptions -->
      <div class="model-legend">
        <div class="legend-item">
          <span class="model-pill model-badge-1">Model 1</span>
          <span>First PPO agent vs fixed bots. Fast, learns legal play well.</span>
        </div>
        <div class="legend-item">
          <span class="model-pill model-badge-2">Model 2</span>
          <span>BC warm-start + PPO. Best high-card laat discipline. Strongest 1v1.</span>
        </div>
        <div class="legend-item">
          <span class="model-pill model-badge-3">Model 3</span>
          <span>League training. Most robust, best against diverse opponents.</span>
        </div>
      </div>

      <div class="rl-setup-footer">
        <button class="primary rl-start-btn" on:click={startRlGame} disabled={rlRunning}>
          {rlRunning ? "Loading…" : "▶ Start RL Game"}
        </button>
        {#if rlMessage}
          <span class="rl-status-msg">{rlMessage}</span>
        {/if}
      </div>
    </section>
  {/if}

  <!-- ====================================================================
       Active RL Banner
       ==================================================================== -->
  {#if rlMode}
    <section class="agent-banner" class:agent-banner-human-turn={isHumanTurnInRl}>
      <div class="banner-left">
        <strong>
          {#if isHumanTurnInRl}
            🧑 Your Turn
          {:else if rlRunning}
            ⚙️ Agent Thinking…
          {:else}
            🤖 RL Game Active
          {/if}
        </strong>
        <span>{rlMessage}</span>
      </div>
      <div class="banner-assignments">
        {#each Array.from({ length: playerCount }, (_, i) => i) as pid}
          <span class="assignment-chip {modelBadgeClass(playerModelSelections[pid])}">
            P{pid}: {rlPlayerModelLabels[String(pid)] ?? playerModelSelections[pid] === 'human' ? 'Human' : 'Bot'}
          </span>
        {/each}
      </div>
      {#if rlLastAction}
        <span class="last-action-tag">
          ↳ {rlLastAction.actor} played <b>{rlLastAction.card}</b>
        </span>
      {/if}
      <!-- RL Controls in banner -->
      <div class="banner-controls">
        <button
          class="primary secondary banner-ctrl-btn"
          on:click={stepRlGame}
          disabled={rlRunning || displayView.gameOver || isHumanTurnInRl}
          title="Step one agent move"
        >⏭ Step</button>
        {#if rlAutoMode}
          <button
            class="primary banner-ctrl-btn banner-stop-btn"
            on:click={stopAutoRl}
            disabled={displayView.gameOver}
            title="Stop auto-play"
          >⏹ Stop Auto</button>
        {:else}
          <button
            class="primary banner-ctrl-btn banner-auto-btn"
            on:click={autoplayRlGame}
            disabled={rlRunning || displayView.gameOver || isHumanTurnInRl}
            title="Auto-play agents (pauses on your turn)"
          >▶▶ Auto</button>
        {/if}
      </div>
    </section>
  {/if}

  {#if errorMessage}
    <div class="alert">{errorMessage}</div>
  {/if}

  <!-- ====================================================================
       Game result banner
       ==================================================================== -->
  {#if displayView.gameOver}
    <div class="result-band">
      <div>
        <span>Winner{displayView.winnerIds.length > 1 ? "s" : ""}</span>
        <strong>{displayView.winnerIds.map(id => displayView.players[id]?.name ?? `P${id}`).join(", ")}</strong>
      </div>
      <div>
        <span>Loser{displayView.loserIds.length > 1 ? "s" : ""}</span>
        <strong>{displayView.loserIds.map(id => displayView.players[id]?.name ?? `P${id}`).join(", ")}</strong>
      </div>
    </div>
  {/if}

  <!-- ====================================================================
       Main board
       ==================================================================== -->
  <div class="game-container">
    <div class="table-col">
      <div class="casino-table">
        <div class="felt">
          {#each displayView.players as player}
            <div
              class="player-avatar pos-{player.id}"
              class:active={player.id === displayView.currentPlayer}
            >
              {#if player.id !== 0}
                <img
                  src="/assets/bot_avatar_{player.id}.png"
                  alt="Avatar for {player.name}"
                  class="avatar-image"
                />
              {/if}
              <div class="avatar-info">
                <strong>{player.name}</strong>
                <span>{player.handCount} cards</span>
                <!-- Show which model this player uses -->
                {#if rlMode}
                  <span class="avatar-model-badge {modelBadgeClass(playerModelSelections[player.id])}">
                    {rlPlayerModelLabels[String(player.id)]?.split(" – ")[0] ?? "Bot"}
                  </span>
                {/if}
                <p>{suitFailureText(player.suitFailures)}</p>
              </div>
              {#if player.hand && player.id !== 0}
                <div
                  class="mini-hand"
                  style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: center; margin-top: 8px;"
                >
                  {#each sortCards(player.hand) as cardId}
                    <span
                      class={cardClass(cardId)}
                      style="background: white; padding: 2px 4px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; border: 1px solid #ccc;"
                    >{formatCard(cardId)}</span>
                  {/each}
                </div>
              {:else if player.id !== 0 && player.handCount > 0}
                <div class="hidden-hand">
                  {#each Array.from({ length: player.handCount }) as _}
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
                  <span>{displayView.players[entry.playerId]?.name}</span>
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

      <!-- ================================================================
           Hand section (Player 0) — directly below table, no scroll
           ================================================================ -->
      <section class="hand-section">
        <div class="hand-heading">
          <div>
            <p class="eyebrow">
              {player0IsModel ? `RL Agent (${MODEL_SHORT[playerModelSelections[0]]})` : "Your hand"}
            </p>
            <h2>{humanHand.length} cards</h2>
          </div>
          {#if isProcessing}
            <p class="thinking">Bots are playing…</p>
          {:else if player0IsModel && rlMode && playerModelSelections[0] !== "human"}
            <p class="thinking">RL agent is in control…</p>
          {:else if isRlAgentThinking}
            <p class="thinking">Bots are playing…</p>
          {:else if isHumanTurnInRl}
            <p>Your turn (RL Mode)!</p>
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
              class:disabled={!legalMoves.includes(cardId) || displayView.gameOver || isProcessing || (rlMode && !isHumanTurnInRl)}
              disabled={!legalMoves.includes(cardId) || displayView.gameOver || isProcessing || (rlMode && !isHumanTurnInRl)}
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

      <!-- ================================================================
           Scoreboard
           ================================================================ -->
      <div class="scoreboard-panel">
        <div class="scoreboard-header">
          <h2>Session scoreboard</h2>
          <div class="scoreboard-meta">
            <span class="games-played-chip">{sessionStats.roundsPlayed} round{sessionStats.roundsPlayed === 1 ? '' : 's'} played</span>
            <button class="reset-stats-btn" on:click={resetStats} title="Reset scoreboard">↺</button>
          </div>
        </div>

        {#if sessionStats.roundsPlayed === 0}
          <p class="scoreboard-empty">Play a game to see results here.</p>
        {:else}
          <div class="scoreboard-rows">
            {#each activePlayerStats as stat, pid}
              {@const winPct = sessionStats.roundsPlayed > 0 ? Math.round(((stat.ranks[1] || 0) / sessionStats.roundsPlayed) * 100) : 0}
              <div class="scoreboard-row">
                <div class="scoreboard-row-top">
                  <span class="scoreboard-name">
                    {#if rlMode}
                      <span class="sb-model-dot {MODEL_COLORS[playerModelSelections[pid]]}"></span>
                    {/if}
                    {stat.name}
                  </span>
                  <span class="scoreboard-record">
                    {#each [1, 2, 3] as rank}
                      {#if stat.ranks[rank]}
                        <span class="rank-count" title="{rank === 1 ? '1st' : rank === 2 ? '2nd' : '3rd'} place">
                          {rank === 1 ? '🥇' : rank === 2 ? '🥈' : '🥉'} {stat.ranks[rank]}
                        </span>
                      {/if}
                    {/each}
                    {#if (stat.ranks[4] || 0) + (stat.ranks[5] || 0) + (stat.ranks[6] || 0) > 0}
                       <span class="rank-count">
                         ... { (stat.ranks[4] || 0) + (stat.ranks[5] || 0) + (stat.ranks[6] || 0) }
                       </span>
                    {/if}
                  </span>
                </div>
                <div class="scoreboard-bar-track" title="1st Place {winPct}%">
                  <div
                    class="scoreboard-bar-fill"
                    style="width: {winPct}%; background: {pid === 0 ? '#10b981' : pid === 1 ? '#3b82f6' : pid === 2 ? '#a855f7' : pid === 3 ? '#f59e0b' : '#64748b'};"
                  ></div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
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
</main>
