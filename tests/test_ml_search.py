"""Pure-numpy ranking tests. These run in CI without the ml extra installed."""

import sys

import numpy as np
import pytest

from hashline.ml.search import (
    cosine_similarity,
    fuse_rankings,
    normalize_rows,
    rank_by_similarity,
)


def test_module_does_not_drag_in_a_model_runtime() -> None:
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


class TestNormalizeRows:
    def test_rows_become_unit_length(self) -> None:
        result = normalize_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_direction_is_kept(self) -> None:
        result = normalize_rows(np.array([[3.0, 4.0]]))
        assert np.allclose(result, [[0.6, 0.8]])

    def test_zero_row_stays_zero_instead_of_nan(self) -> None:
        result = normalize_rows(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert np.allclose(result[0], [0.0, 0.0])

    def test_returns_float32(self) -> None:
        assert normalize_rows(np.array([[1.0, 2.0]], dtype=np.float64)).dtype == (
            np.float32
        )

    def test_rejects_a_non_matrix(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            normalize_rows(np.array([1.0, 2.0]))


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        scores = cosine_similarity(np.array([1.0, 0.0]), np.array([[2.0, 0.0]]))
        assert np.allclose(scores, [1.0])

    def test_orthogonal_vectors_score_zero(self) -> None:
        scores = cosine_similarity(np.array([1.0, 0.0]), np.array([[0.0, 5.0]]))
        assert np.allclose(scores, [0.0])

    def test_opposite_vectors_score_minus_one(self) -> None:
        scores = cosine_similarity(np.array([1.0, 0.0]), np.array([[-3.0, 0.0]]))
        assert np.allclose(scores, [-1.0])

    def test_magnitude_does_not_matter(self) -> None:
        matrix = np.array([[1.0, 1.0], [100.0, 100.0]])
        scores = cosine_similarity(np.array([1.0, 1.0]), matrix)
        assert np.allclose(scores[0], scores[1])

    def test_one_score_per_row_in_row_order(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        scores = cosine_similarity(np.array([1.0, 0.0]), matrix)
        assert np.allclose(scores, [1.0, 0.0, -1.0])

    def test_zero_query_scores_zero_rather_than_nan(self) -> None:
        scores = cosine_similarity(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))
        assert np.allclose(scores, [0.0])

    def test_empty_matrix_yields_no_scores(self) -> None:
        scores = cosine_similarity(np.array([1.0, 0.0]), np.zeros((0, 2)))
        assert scores.shape == (0,)

    def test_rejects_a_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(np.array([1.0, 0.0]), np.array([[1.0, 0.0, 0.0]]))

    def test_rejects_a_non_matrix(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            cosine_similarity(np.array([1.0]), np.array([1.0]))


class TestRankBySimilarity:
    def test_orders_best_first(self) -> None:
        matrix = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
        ranked = rank_by_similarity(np.array([1.0, 0.0]), [10, 20, 30], matrix)
        assert [identifier for identifier, _ in ranked] == [20, 30, 10]

    def test_reports_the_similarity(self) -> None:
        ranked = rank_by_similarity(
            np.array([1.0, 0.0]), [10], np.array([[1.0, 0.0]])
        )
        assert ranked[0][1] == pytest.approx(1.0)

    def test_ties_break_on_the_smaller_id(self) -> None:
        matrix = np.array([[1.0, 0.0], [1.0, 0.0]])
        ranked = rank_by_similarity(np.array([1.0, 0.0]), [30, 7], matrix)
        assert [identifier for identifier, _ in ranked] == [7, 30]

    def test_honours_limit(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ranked = rank_by_similarity(
            np.array([1.0, 0.0]), [1, 2, 3], matrix, limit=2
        )
        assert len(ranked) == 2

    def test_honours_min_score(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ranked = rank_by_similarity(
            np.array([1.0, 0.0]), [1, 2, 3], matrix, min_score=0.5
        )
        assert [identifier for identifier, _ in ranked] == [1]

    def test_empty_input(self) -> None:
        assert rank_by_similarity(np.array([1.0, 0.0]), [], np.zeros((0, 2))) == []

    def test_rejects_a_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ids for"):
            rank_by_similarity(
                np.array([1.0, 0.0]), [1, 2], np.array([[1.0, 0.0]])
            )


class TestFuseRankings:
    def test_an_id_ranked_well_by_both_wins(self) -> None:
        fused = fuse_rankings([[1, 2, 3], [1, 3, 2]])
        assert [identifier for identifier, _ in fused][0] == 1

    def test_appearing_in_both_lists_beats_leading_only_one(self) -> None:
        keyword = [99, 1]
        semantic = [7, 1]
        fused = [identifier for identifier, _ in fuse_rankings([keyword, semantic])]
        assert fused[0] == 1

    def test_a_first_place_still_outweighs_a_lower_pair(self) -> None:
        # 99 leads the keyword list and trails the semantic one; 2 is last then
        # second. The single first place wins -- fusion blends, it does not
        # simply reward agreement.
        ranked = fuse_rankings([[99, 1, 2], [1, 2, 99]])
        assert [identifier for identifier, _ in ranked] == [1, 99, 2]

    def test_keeps_ids_that_appear_in_only_one_list(self) -> None:
        fused = fuse_rankings([[1], [2]])
        assert {identifier for identifier, _ in fused} == {1, 2}

    def test_ties_break_on_the_smaller_id(self) -> None:
        assert [identifier for identifier, _ in fuse_rankings([[5], [3]])] == [3, 5]

    def test_scores_are_the_reciprocal_ranks(self) -> None:
        fused = fuse_rankings([[7]], k=60)
        assert fused[0][1] == pytest.approx(1 / 61)

    def test_a_smaller_k_sharpens_the_top_of_the_list(self) -> None:
        gentle = dict(fuse_rankings([[1, 2]], k=60))
        sharp = dict(fuse_rankings([[1, 2]], k=1))
        assert sharp[1] - sharp[2] > gentle[1] - gentle[2]

    def test_honours_limit(self) -> None:
        assert len(fuse_rankings([[1, 2, 3]], limit=2)) == 2

    def test_no_rankings(self) -> None:
        assert fuse_rankings([]) == []

    def test_rejects_a_k_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            fuse_rankings([[1]], k=0)
