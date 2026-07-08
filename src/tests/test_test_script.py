import unittest

import test


class DummyAgent:
    def __init__(self):
        self.loaded_checkpoint = None
        self.loaded_weights = None

    def load_checkpoint(self, checkpoint_dir):
        self.loaded_checkpoint = checkpoint_dir
        return f"{checkpoint_dir}/ckpt"

    def load_weights(self, weights_path):
        self.loaded_weights = weights_path


class TestScriptLoading(unittest.TestCase):
    def test_load_agent_weights_prefers_checkpoint_when_available(self):
        agent = DummyAgent()
        restored = test.load_agent_artifacts(agent, weights_path='best_mspacman', checkpoint_dir='checkpoints')

        self.assertEqual(restored, 'checkpoints/ckpt')
        self.assertEqual(agent.loaded_checkpoint, 'checkpoints')
        self.assertIsNone(agent.loaded_weights)


if __name__ == '__main__':
    unittest.main()
