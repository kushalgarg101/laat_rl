import pytest

from rl.laat_game.cards import card_id
from rl.scripts.evaluate_strategy import composite_score, rank_percentile, summarize_bucket


def test_rank_percentile_uses_legal_card_ranks():
    legal = [
        card_id("S", 2),
        card_id("H", 5),
        card_id("D", 10),
        card_id("C", 14),
    ]

    assert rank_percentile(card_id("S", 2), legal) == pytest.approx(0.25)
    assert rank_percentile(card_id("D", 10), legal) == pytest.approx(0.75)
    assert rank_percentile(card_id("C", 14), legal) == pytest.approx(1.0)


def test_rank_percentile_counts_rank_ties_as_upper_tie_position():
    legal = [
        card_id("S", 5),
        card_id("H", 5),
        card_id("D", 9),
        card_id("C", 13),
    ]

    assert rank_percentile(card_id("S", 5), legal) == pytest.approx(0.5)


def test_summarize_bucket_reports_high_and_low_card_rates():
    rows = [
        (14.0, 8.0, 1.0, 4.0),
        (12.0, 8.0, 0.75, 4.0),
        (5.0, 8.0, 0.25, 4.0),
        (2.0, 8.0, 0.1, 4.0),
    ]

    summary = summarize_bucket(rows)

    assert summary.n == 4
    assert summary.avg_rank_percentile == pytest.approx(0.525)
    assert summary.high_card_rate_p75 == pytest.approx(0.5)
    assert summary.low_card_rate_p25 == pytest.approx(0.5)
    assert summary.avg_legal_count == pytest.approx(4.0)


def test_composite_score_rewards_wins_and_penalizes_losses_cards_and_invalid_actions():
    strong = composite_score(win_rate=0.9, loss_rate=0.05, avg_final_hand=2.0, invalid_actions=0)
    weak = composite_score(win_rate=0.6, loss_rate=0.3, avg_final_hand=12.0, invalid_actions=1)

    assert strong > weak
