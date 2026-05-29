import numpy as np
from sb3_contrib import MaskablePPO

from rl.laat_game.cards import card_id
from rl.laat_game.env import LaatCardEnv, OBS_SIZE
from rl.laat_game.engine import GameConfig, GameEvent, GameState, PlayerState, TrickEntry
from rl.laat_game.training_callbacks import WandbPredictionCallback

TRICK_OFFSET = 52
HAND_COUNTS_OFFSET = TRICK_OFFSET + 6 * 52
SUIT_FAILURES_OFFSET = HAND_COUNTS_OFFSET + 6
CURRENT_PLAYER_OFFSET = SUIT_FAILURES_OFFSET + 6 * 4
LEAD_SUIT_OFFSET = CURRENT_PLAYER_OFFSET + 6
HIGHEST_PLAYER_OFFSET = LEAD_SUIT_OFFSET + 5


def test_env_reset_observation_and_mask_shapes():
    env = LaatCardEnv(player_count=4, seed=5)
    obs, info = env.reset(seed=5)
    mask = env.action_masks()
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert mask.shape == (52,)
    assert mask.dtype == bool
    assert mask.any()
    assert "hand_counts" in info


def test_env_runs_masked_steps_without_invalid_actions():
    env = LaatCardEnv(player_count=4, max_rounds=2, max_actions_per_game=400, seed=9)
    env.reset(seed=9)
    done = False
    guard = 0
    while not done and guard < 100:
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        _, _, terminated, truncated, info = env.step(action)
        assert not info.get("invalid_action", False)
        assert info["safety_errors"] == []
        done = terminated or truncated
        guard += 1
    assert guard > 0


def test_unseeded_resets_vary_but_seed_sequence_is_reproducible():
    env_a = LaatCardEnv(player_count=4, seed=17)
    env_b = LaatCardEnv(player_count=4, seed=17)

    seq_a = []
    seq_b = []
    for _ in range(3):
        env_a.reset()
        env_b.reset()
        seq_a.append(tuple(env_a.state.players[0].hand))
        seq_b.append(tuple(env_b.state.players[0].hand))

    assert len(set(seq_a)) > 1
    assert seq_a == seq_b


def test_observation_encodes_trick_ownership_and_current_highest_player():
    env = LaatCardEnv(player_count=2)
    env.state = manual_state(
        hands=[[card_id("H", 2)], [card_id("S", 9)]],
        current_player=1,
        trick=[(0, card_id("S", 5)), (1, card_id("S", 7))],
        lead_suit="S",
    )

    obs = env.current_observation()
    assert obs[TRICK_OFFSET + 0 * 52 + card_id("S", 5)] == 1.0
    assert obs[TRICK_OFFSET + 1 * 52 + card_id("S", 7)] == 1.0
    assert obs[HIGHEST_PLAYER_OFFSET + 1] == 1.0


def test_reward_penalizes_agent_laat_collection():
    env = LaatCardEnv(player_count=2)
    env.state = manual_state(hands=[[card_id("S", 5), card_id("H", 2)], []], current_player=0)
    event = GameEvent(
        "laat",
        "P0 collects",
        1,
        1,
        {"collector_id": 0, "card_count": 2},
    )
    assert env._reward(before_agent=1, new_events=[event]) < 0


def test_reward_credits_opponent_laat_collection():
    env = LaatCardEnv(player_count=2)
    env.state = manual_state(hands=[[card_id("H", 2)], [card_id("S", 5), card_id("D", 3)]], current_player=0)
    event = GameEvent(
        "laat",
        "P1 collects",
        1,
        1,
        {"collector_id": 1, "card_count": 2},
    )
    assert env._reward(before_agent=2, new_events=[event]) > 0


def test_wandb_callback_resolves_base_env_through_sb3_wrappers():
    env = LaatCardEnv(seed=7)
    env.reset(seed=7)
    model = MaskablePPO("MlpPolicy", env, n_steps=32, batch_size=16, device="cpu")
    callback = WandbPredictionCallback()
    callback.init_callback(model)
    base_env = callback._base_env()
    assert isinstance(base_env, LaatCardEnv)
    assert callback._top_predictions(base_env, base_env.action_masks())


def manual_state(hands, current_player, trick=None, lead_suit=None):
    players = [PlayerState(i, f"P{i}", list(hand)) for i, hand in enumerate(hands)]
    return GameState(
        config=GameConfig(player_count=len(players), max_rounds=10),
        players=players,
        current_player=current_player,
        current_trick=[TrickEntry(player_id, card) for player_id, card in trick or []],
        lead_suit=lead_suit,
        suit_failures=[[False for _ in range(4)] for _ in players],
        first_move_card_id=None,
    )
