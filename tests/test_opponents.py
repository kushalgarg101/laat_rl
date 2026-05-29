import numpy as np

from rl.laat_game.cards import card_id
from rl.laat_game.engine import GameConfig, GameState, PlayerState, TrickEntry, get_legal_moves
from rl.laat_game.env import LaatCardEnv
from rl.laat_game.observation import encode_observation
from rl.laat_game.opponents import build_opponent_policy


CURRENT_PLAYER_OFFSET = 52 + 6 * 52 + 6 + 6 * 4


def test_league_opponent_only_returns_legal_actions():
    env = LaatCardEnv(player_count=4, seed=3, opponent_pool="league")
    env.reset(seed=3)

    for _ in range(25):
        if env.state.game_over:
            break
        if env.state.current_player == 0:
            action = int(np.flatnonzero(env.action_masks())[0])
        else:
            legal = get_legal_moves(env.state)
            action = env.opponent_policy.choose_action(env.state)
            assert action in legal
        env.step(action) if env.state.current_player == 0 else None


def test_perspective_observation_maps_current_player_to_zero_for_acting_player():
    state = manual_state(
        hands=[
            [card_id("S", 2)],
            [card_id("H", 3)],
            [card_id("D", 4)],
        ],
        current_player=2,
        trick=[(1, card_id("S", 9))],
        lead_suit="S",
    )

    obs = encode_observation(state, perspective_player=2)

    assert obs[card_id("D", 4)] == 1.0
    assert obs[CURRENT_PLAYER_OFFSET + 0] == 1.0


def test_build_opponent_policy_supports_named_pools():
    state = manual_state(
        hands=[[card_id("S", 2)], [card_id("S", 3), card_id("H", 14)]],
        current_player=1,
        trick=[(0, card_id("S", 7))],
        lead_suit="S",
    )

    for pool in ["baseline", "random", "low_card", "high_card", "safe_suit", "pressure_suit", "league"]:
        policy = build_opponent_policy(pool, seed=7)
        policy.start_episode(state)
        assert policy.choose_action(state) in get_legal_moves(state)


def test_env_reset_reuses_checkpoint_opponent_policy(monkeypatch, tmp_path):
    import rl.laat_game.opponents as opponents

    load_count = 0

    class FakeCheckpointPolicy(opponents.BaseOpponentPolicy):
        def __init__(self, checkpoint_path, device):
            nonlocal load_count
            load_count += 1

        def choose_action(self, state):
            return get_legal_moves(state)[0]

    monkeypatch.setattr(opponents, "CheckpointPolicy", FakeCheckpointPolicy)
    checkpoint = tmp_path / "fake_checkpoint.zip"
    checkpoint.write_text("fake", encoding="utf-8")
    env = LaatCardEnv(
        player_count=4,
        seed=5,
        opponent_pool="league",
        opponent_checkpoint=checkpoint,
    )

    env.reset(seed=5)
    env.reset(seed=6)

    assert load_count == 1


def test_env_marks_game_over_when_bot_advance_hits_guard():
    env = LaatCardEnv(player_count=4, seed=2, opponent_pool="baseline", max_actions_per_game=1)
    env.reset(seed=2)

    assert env.state.game_over or env.state.current_player == 0


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
