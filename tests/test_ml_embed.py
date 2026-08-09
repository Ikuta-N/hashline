"""Codec and availability tests run everywhere.

Anything that needs a downloaded model is marked ``slow`` and excluded by
default, so CI never fetches one.
"""

import sys

import numpy as np
import pytest

from hashline.ml.embed import (
    DEFAULT_MODEL,
    MlExtraNotInstalled,
    embed_texts,
    is_available,
    load_model,
    pack_vector,
    unpack_matrix,
    unpack_vector,
)
from hashline.ml.protocols import Embedder
from hashline.ml.search import rank_by_similarity


def test_importing_the_module_loads_no_model_runtime() -> None:
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


class TestIsAvailable:
    def test_reports_a_bool(self) -> None:
        assert isinstance(is_available(), bool)

    def test_agrees_with_load_model(self) -> None:
        if is_available():
            return
        with pytest.raises(MlExtraNotInstalled, match="ml"):
            load_model()


class TestVectorCodec:
    def test_round_trips_a_vector(self) -> None:
        vector = np.array([0.5, -1.25, 3.0], dtype=np.float32)
        assert np.array_equal(unpack_vector(pack_vector(vector)), vector)

    def test_narrows_float64_to_float32(self) -> None:
        packed = pack_vector(np.array([1.0, 2.0], dtype=np.float64))
        assert unpack_vector(packed).dtype == np.float32
        assert len(packed) == 8

    def test_flattens_a_single_row(self) -> None:
        packed = pack_vector(np.array([[1.0, 2.0, 3.0]]))
        assert unpack_vector(packed).shape == (3,)

    def test_unpacked_vector_is_writable(self) -> None:
        vector = unpack_vector(pack_vector(np.array([1.0, 2.0])))
        vector[0] = 9.0  # a frombuffer view would raise here
        assert vector[0] == 9.0

    def test_rejects_a_truncated_blob(self) -> None:
        with pytest.raises(ValueError, match="not a float32 vector"):
            unpack_vector(b"\x00\x00\x00")

    def test_empty_blob_gives_an_empty_vector(self) -> None:
        assert unpack_vector(b"").shape == (0,)

    def test_the_stored_bytes_are_little_endian(self) -> None:
        """The format is fixed, not inherited from whoever wrote the row.

        A .db file moves between machines, so a natively packed vector read
        on a host of the other byte order would come back as plausible
        garbage -- no exception, only wrong rankings.
        """
        # float32 1.0 is 0x3F800000, so little-endian puts 0x3F last.
        assert pack_vector(np.array([1.0], dtype=np.float32)) == b"\x00\x00\x80\x3f"

    def test_reads_a_blob_written_on_a_machine_of_either_byte_order(self) -> None:
        vector = np.array([0.5, -1.25, 3.0], dtype=np.float32)
        written_big_endian = vector.astype(">f4").tobytes()
        assert not np.array_equal(unpack_vector(written_big_endian), vector), (
            "the fixture is not actually byte-swapped"
        )
        assert np.array_equal(unpack_vector(vector.astype("<f4").tobytes()), vector)

    def test_a_dimension_short_of_the_row_is_refused(self) -> None:
        """embeddings.dim is the record of how wide the vector should be.

        Without the cross-check a blob that lost dimensions reads back as a
        shorter, entirely plausible vector, and unpack_matrix would only
        notice if some other row happened to disagree with it.
        """
        packed = pack_vector(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert unpack_vector(packed, expected_dim=3).shape == (3,)
        with pytest.raises(ValueError, match="the row records 4"):
            unpack_vector(packed, expected_dim=4)


class TestUnpackMatrix:
    def test_stacks_rows_in_order(self) -> None:
        blobs = [
            pack_vector(np.array([1.0, 0.0])),
            pack_vector(np.array([0.0, 1.0])),
        ]
        assert np.array_equal(unpack_matrix(blobs), np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_feeds_the_ranker(self) -> None:
        blobs = [
            pack_vector(np.array([0.0, 1.0])),
            pack_vector(np.array([1.0, 0.0])),
        ]
        ranked = rank_by_similarity(
            np.array([1.0, 0.0]), [10, 20], unpack_matrix(blobs)
        )
        assert [identifier for identifier, _ in ranked] == [20, 10]

    def test_rejects_mixed_dimensions(self) -> None:
        blobs = [pack_vector(np.array([1.0])), pack_vector(np.array([1.0, 2.0]))]
        with pytest.raises(ValueError, match="mixed dimensions"):
            unpack_matrix(blobs)

    def test_no_blobs(self) -> None:
        assert unpack_matrix([]).shape == (0, 0)


class TestEmbedTexts:
    def test_empty_input_needs_no_model(self) -> None:
        assert embed_texts([]).shape == (0, 0)

    def test_without_the_extra_it_says_so(self) -> None:
        if is_available():
            pytest.skip("the ml extra is installed")
        with pytest.raises(MlExtraNotInstalled):
            embed_texts(["anything"])

    def test_uses_a_model_it_is_given(self) -> None:
        class FakeEncoder:
            def encode(self, texts: list[str], **_: object) -> np.ndarray:
                return np.array([[float(len(text)), 0.0] for text in texts])

        result = embed_texts(["ab", "abcd"], model=FakeEncoder())
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        assert result[1][0] == 4.0

    def test_an_embedder_needs_nothing_beyond_the_protocol(self) -> None:
        """The one method in Embedder is the whole contract.

        embed_texts used to pass convert_to_numpy=True, so anything standing
        in for a model had to accept sentence-transformers' keyword arguments
        as well. That option now lives in the adapter load_model returns.
        """

        class MinimalEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[float(len(text))] for text in texts])

        embedder: Embedder = MinimalEmbedder()
        assert embed_texts(["abc"], model=embedder)[0][0] == 3.0


@pytest.mark.slow
class TestAgainstARealModel:
    """Downloads a model. Excluded by default; never run in CI."""

    def test_embeds_to_a_stable_width(self) -> None:
        vectors = embed_texts(["a note about databases", "別の話題"])
        assert vectors.shape[0] == 2
        assert vectors.shape[1] > 0
        assert vectors.dtype == np.float32

    def test_related_text_outranks_unrelated_text(self) -> None:
        model = load_model(DEFAULT_MODEL)
        notes = [
            "the cat sat on the mat",
            "SQLite full-text search with BM25 ranking",
        ]
        matrix = embed_texts(notes, model=model)
        query = embed_texts(["database search index"], model=model)[0]
        ranked = rank_by_similarity(query, [0, 1], matrix)
        assert ranked[0][0] == 1

    def test_a_vector_survives_the_codec(self) -> None:
        vector = embed_texts(["round trip"])[0]
        assert np.allclose(unpack_vector(pack_vector(vector)), vector)
