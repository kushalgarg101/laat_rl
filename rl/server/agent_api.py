from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import torch

from rl.laat_game.cards import format_card, sort_cards
from rl.laat_game.engine import GameConfig, GameState, apply_move, choose_bot_move, create_game, get_legal_moves, run_safety_checks
from rl.laat_game.env import LaatCardEnv
from rl.laat_game.observation import encode_observation, action_mask as build_action_mask


# ---------------------------------------------------------------------------
# Known model catalogue exposed to the UI
# ---------------------------------------------------------------------------

MODEL_CATALOGUE = [
    {
        "id": "model1",
        "label": "Model 1 – Masked PPO",
        "description": "First RL agent, trained against fixed heuristic bots. Win rate ~84%.",
        "path": "models/maskable_ppo_laat/latest.zip",
    },
    {
        "id": "model2",
        "label": "Model 2 – BC + PPO",
        "description": "Behavior-cloned warm-start then PPO fine-tuned. Excellent high-card laat discipline. Win rate ~95%.",
        "path": "models/model2_ppo/latest.zip",
    },
    {
        "id": "model3",
        "label": "Model 3 – League PPO",
        "description": "Trained against a diverse league of opponents. Best overall robustness. Win rate ~90%.",
        "path": "models/model3_ppo/checkpoints/laat_800000_steps.zip",
    },
]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NewSessionRequest(BaseModel):
    # player_models: list of model IDs or paths, one per player slot.
    # None / empty string means use the heuristic bot for that slot.
    player_models: list[str | None] = []
    player_count: int = 4
    max_rounds: int = 12
    hand_threshold: int = 52
    seed: int | None = None
    reveal_hands: bool = False
    device: str = "cuda"


class StepRequest(BaseModel):
    session_id: str
    reveal_hands: bool = False
    cardId: int | None = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class AgentSession:
    def __init__(self, request: NewSessionRequest) -> None:
        device = request.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # Build a mapping from player_id -> MaskablePPO (or None for heuristic).
        self.player_models: dict[int, MaskablePPO | None] = {}
        self.player_model_labels: dict[int, str] = {}

        # Resolve catalogue IDs to paths
        catalogue_by_id = {m["id"]: m for m in MODEL_CATALOGUE}
        catalogue_by_path = {m["path"]: m for m in MODEL_CATALOGUE}

        player_model_specs = list(request.player_models or [])
        # Pad / truncate to player_count
        while len(player_model_specs) < request.player_count:
            player_model_specs.append(None)
        player_model_specs = player_model_specs[: request.player_count]

        loaded_cache: dict[str, MaskablePPO] = {}

        for pid, spec in enumerate(player_model_specs):
            if spec == "human":
                self.player_models[pid] = None
                self.player_model_labels[pid] = "Human"
                continue

            if not spec:
                self.player_models[pid] = None
                self.player_model_labels[pid] = "Heuristic Bot"
                continue

            # Allow catalogue ID ("model1") or raw path
            if spec in catalogue_by_id:
                path_str = catalogue_by_id[spec]["path"]
                label = catalogue_by_id[spec]["label"]
            else:
                path_str = spec
                label = catalogue_by_path.get(spec, {}).get("label", Path(spec).stem)

            model_path = Path(path_str)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found for player {pid}: {model_path}")

            if path_str not in loaded_cache:
                # Create a minimal env for SB3 to use when loading
                env = LaatCardEnv(
                    player_count=request.player_count,
                    max_rounds=request.max_rounds,
                    hand_threshold=request.hand_threshold,
                    seed=request.seed,
                )
                loaded_cache[path_str] = MaskablePPO.load(model_path, env=env, device=device)

            self.player_models[pid] = loaded_cache[path_str]
            self.player_model_labels[pid] = label

        # The canonical game state
        self.state = create_game(
            GameConfig(
                player_count=request.player_count,
                max_rounds=request.max_rounds,
                hand_threshold=request.hand_threshold,
                seed=request.seed,
            )
        )

        # We still keep an env handy for observation encoding for player 0.
        # For other players we encode observations manually with perspective rotation.
        self.env = LaatCardEnv(
            player_count=request.player_count,
            max_rounds=request.max_rounds,
            hand_threshold=request.hand_threshold,
            seed=request.seed,
        )
        self.env.state = self.state

        self.last_action: dict | None = None

    def choose_action(self, state: GameState, player_id: int) -> int:
        """Return the card ID for the current player, using their assigned model or heuristic."""
        model = self.player_models.get(player_id)
        if model is None:
            return choose_bot_move(state)

        # Encode observation from this player's perspective
        obs = encode_observation(state, perspective_player=player_id)
        mask = build_action_mask(state, player_id)

        import numpy as np
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Laat Card Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, AgentSession] = {}


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/api/models")
def list_models():
    """Return the list of known models with existence check."""
    result = []
    for m in MODEL_CATALOGUE:
        entry = dict(m)
        entry["available"] = Path(m["path"]).exists()
        result.append(entry)
    return {"models": result}


@app.post("/api/agent-game/new")
def new_session(request: NewSessionRequest):
    try:
        session = AgentSession(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = str(uuid4())
    sessions[session_id] = session

    player_labels = {str(pid): label for pid, label in session.player_model_labels.items()}

    return {
        "sessionId": session_id,
        "playerModelLabels": player_labels,
        "device": session.device,
        "view": public_view(session.state, request.reveal_hands),
        "lastAction": session.last_action,
    }


@app.post("/api/agent-game/step")
def step_session(request: StepRequest):
    session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = session.state
    session.last_action = None

    if not state.game_over:
        current_pid = state.current_player

        is_human = session.player_model_labels.get(current_pid) == "Human"
        if is_human:
            if request.cardId is None:
                raise HTTPException(status_code=400, detail="A cardId must be provided for a human player's turn.")
            card_id = request.cardId
            actor_label = state.players[current_pid].name
        else:
            card_id = session.choose_action(state, current_pid)
            actor_label = session.player_model_labels.get(current_pid, state.players[current_pid].name)
            if session.player_models.get(current_pid) is not None:
                actor_label = f"{state.players[current_pid].name} ({actor_label})"

        legal = get_legal_moves(state)
        if card_id not in legal:
            raise HTTPException(status_code=400, detail=f"Predicted/Requested illegal action {card_id}; legal={legal}")

        apply_move(state, card_id)
        session.env.state = state

        session.last_action = {
            "playerId": current_pid,
            "actor": actor_label,
            "cardId": card_id,
            "card": format_card(card_id),
        }

    player_labels = {str(pid): label for pid, label in session.player_model_labels.items()}

    return {
        "sessionId": request.session_id,
        "playerModelLabels": player_labels,
        "device": session.device,
        "view": public_view(state, request.reveal_hands),
        "lastAction": session.last_action,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def public_view(state: GameState, reveal_hands: bool) -> dict:
    return {
        "currentPlayer": state.current_player,
        "currentTrick": [{"playerId": entry.player_id, "cardId": entry.card_id} for entry in state.current_trick],
        "leadSuit": state.lead_suit,
        "roundDiscardCount": len(state.round_discard),
        "round": state.round,
        "turn": state.turn,
        "gameOver": state.game_over,
        "winnerIds": list(state.winner_ids),
        "loserIds": list(state.loser_ids),
        "players": [
            {
                "id": player.id,
                "name": player.name,
                "handCount": len(player.hand),
                "hand": sort_cards(player.hand) if reveal_hands else (sort_cards(player.hand) if player.id == 0 else None),
                "suitFailures": list(state.suit_failures[player.id]),
            }
            for player in state.players
        ],
        "events": [event_to_json(event) for event in state.events[-80:]],
        "safetyErrors": run_safety_checks(state),
        "round_rankings": state.round_rankings,
    }


def event_to_json(event) -> dict:
    data = asdict(event)
    return {
        "round": data["round"],
        "turn": data["turn"],
        "type": data["type"],
        "message": data["message"],
        "metadata": data.get("metadata", {}),
    }
