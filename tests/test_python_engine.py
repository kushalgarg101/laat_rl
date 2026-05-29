from rl.laat_game.cards import STARTING_CARD_ID, card_id, create_deck
from rl.laat_game.engine import GameConfig, GameState, PlayerState, TrickEntry, apply_move, create_game, get_legal_moves, run_safety_checks


def test_deck_has_52_unique_cards():
    deck = create_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_cyclic_deal_for_six_players_preserves_all_cards():
    state = create_game(GameConfig(player_count=6, seed=1))
    counts = [len(player.hand) for player in state.players]
    assert counts == [9, 9, 9, 9, 8, 8]
    assert run_safety_checks(state) == []


def test_ace_of_spades_holder_starts_with_only_starting_card_legal():
    state = create_game(GameConfig(player_count=4, seed=3))
    assert STARTING_CARD_ID in state.players[state.current_player].hand
    assert get_legal_moves(state) == [STARTING_CARD_ID]


def test_follow_suit_is_required_when_player_has_lead_suit():
    state = manual_state(
        hands=[
            [card_id("S", 5)],
            [card_id("S", 7), card_id("H", 2)],
        ],
        current_player=1,
        trick=[(0, card_id("S", 5))],
        lead_suit="S",
    )
    assert get_legal_moves(state) == [card_id("S", 7)]


def test_laat_allows_any_card_when_player_lacks_lead_suit():
    state = manual_state(
        hands=[
            [card_id("S", 5)],
            [card_id("H", 2), card_id("D", 9)],
        ],
        current_player=1,
        trick=[(0, card_id("S", 5))],
        lead_suit="S",
    )
    assert get_legal_moves(state) == [card_id("H", 2), card_id("D", 9)]


def test_laat_collection_includes_laat_card():
    state = manual_state(
        hands=[
            [],
            [card_id("H", 2)],
        ],
        current_player=1,
        trick=[(0, card_id("S", 5))],
        lead_suit="S",
    )
    apply_move(state, card_id("H", 2))
    assert sorted(state.players[0].hand) == sorted([card_id("S", 5), card_id("H", 2)])
    assert state.current_player == 0
    assert state.current_trick == []
    assert state.suit_failures[1][0] is True


def test_clean_trick_goes_to_round_discard():
    state = manual_state(
        hands=[
            [card_id("H", 2)],
            [card_id("S", 7), card_id("H", 3)],
        ],
        current_player=1,
        trick=[(0, card_id("S", 5))],
        lead_suit="S",
    )
    apply_move(state, card_id("S", 7))
    assert sorted(state.round_discard) == sorted([card_id("S", 5), card_id("S", 7)])
    assert state.current_player == 1


def test_random_masked_episode_smoke():
    state = create_game(GameConfig(player_count=4, seed=11, max_rounds=2, max_actions_per_game=400))
    guard = 0
    while not state.game_over and guard < 400:
        legal = get_legal_moves(state)
        assert legal
        apply_move(state, legal[0])
        assert run_safety_checks(state) == []
        guard += 1
    assert state.game_over


def manual_state(hands, current_player, trick, lead_suit):
    players = [PlayerState(i, f"P{i}", list(hand)) for i, hand in enumerate(hands)]
    return GameState(
        config=GameConfig(player_count=len(players), max_rounds=10),
        players=players,
        current_player=current_player,
        current_trick=[TrickEntry(player_id, card) for player_id, card in trick],
        lead_suit=lead_suit,
        suit_failures=[[False for _ in range(4)] for _ in players],
        first_move_card_id=None,
    )
