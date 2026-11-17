"""The eval viewer decodes a block's 16-bit heights with the block's own step (maxHeight / 65536), not the anchor's q:
a block whose roads reached far past the anchor's range was built on a coarser step."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import earth_view  # noqa: E402


class BlockQuantumTest(unittest.TestCase):
    def test_its_own_max_height(self):
        self.assertEqual(earth_view.block_quantum({"maxHeight": 4096.0, "q": 0.03125}), 0.0625)

    def test_the_anchors_q_without_one(self):
        self.assertEqual(earth_view.block_quantum({"q": 0.03125}), 0.03125)


if __name__ == "__main__":
    unittest.main()
