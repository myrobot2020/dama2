"""Regression tests for topic search ranking (AN books 4–11 JSON)."""

from __future__ import annotations

import unittest

import topic_search_server as tss


class TestTopicSearchRegression(unittest.TestCase):
    def assert_first_hit(self, query: str, expected_sutta_id: str) -> None:
        out = tss.run_search(query, max_columns=5)
        self.assertTrue(out.get("top"), msg=f"no hits for {query!r}")
        first = out["top"][0].get("sutta_id")
        self.assertEqual(
            first,
            expected_sutta_id,
            msg=f"{query!r}: first hit {first!r}, expected {expected_sutta_id!r}; top={out['top'][:3]!r}",
        )

    def test_advantages_of_loving_kindness(self) -> None:
        self.assert_first_hit("advantages of loving-kindness", "11.16")

    def test_transference_of_merit(self) -> None:
        self.assert_first_hit("transference of merit", "10.177")

    def test_right_efforts(self) -> None:
        self.assert_first_hit("right efforts", "4.2.13")

    def test_fetters(self) -> None:
        self.assert_first_hit("fetters", "10.13")

    def test_objects_of_the_training(self) -> None:
        self.assert_first_hit("objects of the training", "10.31")


if __name__ == "__main__":
    unittest.main()
