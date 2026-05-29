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


class NewSessionRequest(BaseModel):
    model_path: str = "models/maskable_ppo_laat/latest.zip"
    player_count: int = 4
    max_rounds: int = 12
    hand_threshold: int = 52
    seed: int | None = None
    reveal_hands: bool = False
    device: str = "cuda"


class StepRequest(BaseModel):
    session_id: str
    reveal_hands: bool = False


class AgentSession:
    def __init__(self, request: NewSessionRequest) -> None:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        device = request.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.env = LaatCardEnv(
            player_count=request.player_count,
            max_rounds=request.max_rounds,
            hand_threshold=request.hand_threshold,
            seed=request.seed,
        )
        self.state = create_game(
            GameConfig(
                player_count=request.player_count,
                max_rounds=request.max_rounds,
                hand_threshold=request.hand_threshold,
                seed=request.seed,
            )
        )
        self.env.state = self.state
        self.model = MaskablePPO.load(model_path, env=self.env, device=device)
        self.model_path = str(model_path)
        self.device = device
        self.last_action: dict | None = None


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


@app.post("/api/agent-game/new")
def new_session(request: NewSessionRequest):
    try:
        session = AgentSession(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session_id = str(uuid4())
    sessions[session_id] = session
    return {
        "sessionId": session_id,
        "modelPath": session.model_path,
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
        if state.current_player == 0:
            obs = session.env.current_observation()
            action, _ = session.model.predict(obs, action_masks=get_action_masks(session.env), deterministic=True)
            actor = "RL Agent"
            card_id = int(action)
        else:
            card_id = choose_bot_move(state)
            actor = state.players[state.current_player].name

        legal = get_legal_moves(state)
        if card_id not in legal:
            raise HTTPException(status_code=500, detail=f"Predicted illegal action {card_id}; legal={legal}")
        apply_move(state, card_id)
        session.env.state = state
        session.last_action = {
            "playerId": state.events[-1].metadata.get("failed_player_id", state.current_player) if state.events else None,
            "actor": actor,
            "cardId": card_id,
            "card": format_card(card_id),
        }

    return {
        "sessionId": request.session_id,
        "modelPath": session.model_path,
        "device": session.device,
        "view": public_view(state, request.reveal_hands),
        "lastAction": session.last_action,
    }


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
                "name": "RL Agent" if player.id == 0 else player.name,
                "handCount": len(player.hand),
                "hand": sort_cards(player.hand) if player.id == 0 or reveal_hands else None,
                "suitFailures": list(state.suit_failures[player.id]),
            }
            for player in state.players
        ],
        "events": [event_to_json(event) for event in state.events[-80:]],
        "safetyErrors": run_safety_checks(state),
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
