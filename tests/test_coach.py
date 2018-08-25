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
            return (5.2, {'a': 1}), '5|3|1|4'

        network = SimpleFC(name='simple_fc', directory=self.test_dir)
        cch = coach.Coach(network=network)

        cch.train(
            target=coach.targets.epoch_reached(3),
            train_one_epoch=train_one_epoch,
            measure_performance=measure_performance,
            settings_notes={'learning_rate': 5},
            checkpoint_frequency=10,
            checkpoint=-1
        )

        created_checkpoint = network.latest_checkpoint()
        self.assertEqual(created_checkpoint.id, 0)
        self.assertEqual(created_checkpoint.notes, {
            'train_set_performance': (5.2, {'a': 1}),
            'dev_set_performance': '5|3|1|4',
            'learning_rate': 5
        })

if __name__ == '__main__':
    unittest.main()
