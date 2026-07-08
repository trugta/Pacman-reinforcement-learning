import unittest
import numpy as np

from buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_stores_observations_in_uint8(self):
        buffer = ReplayBuffer((84, 84, 4))
        self.assertEqual(buffer.states.dtype, np.uint8)
        self.assertEqual(buffer.next_states.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
