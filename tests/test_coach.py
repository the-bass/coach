import unittest
import os
import shutil
import coach
import coach.targets
from constants import TEST_TMP_DIRECTORY
from fixtures.simple_fc import SimpleFC


class TestCoach(unittest.TestCase):

    def setUp(self):
        self.test_dir = TEST_TMP_DIRECTORY
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(TEST_TMP_DIRECTORY)

    def test_checkpoint_creation(self):
        self.loss = 1
        def train_one_epoch():
            self.loss -= 0.1
            return self.loss

        def measure_performance():
            return 5.2, 6.1

        network = SimpleFC(name='simple_fc', directory=self.test_dir)
        cch = coach.Coach(network=network)

        cch.train(
            target=coach.targets.epoch_reached(3),
            train_one_epoch=train_one_epoch,
            measure_performance=measure_performance,
            notes='LR=0.001',
            checkpoint_frequency=10,
            checkpoint=-1
        )

        created_checkpoint = network.latest_checkpoint()
        self.assertEqual(created_checkpoint.id, 0)
        self.assertEqual(created_checkpoint.notes, 'LR=0.001')
        self.assertEqual(created_checkpoint.train_set_performance, 5.2)
        self.assertEqual(created_checkpoint.dev_set_performance, 6.1)
        self.assertEqual(created_checkpoint.losses_since_last_checkpoint, [0.9, 0.8, 0.7000000000000001])

if __name__ == '__main__':
    unittest.main()
