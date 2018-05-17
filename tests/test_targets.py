import unittest
import coach.targets as targets


class TestTargets(unittest.TestCase):

    def test_loss_stagnation(self):
        training_target = targets.loss_stagnation(
            epsilon=1,
            conservativity=3)

        # Test argument validation.
        with self.assertRaises(ValueError) as context:
            targets.loss_stagnation(
                epsilon=1,
                conservativity=1)
        self.assertEqual(
            "specify a conservativity greater than or equal to 2",
            str(context.exception))

        # Test processing.
        self.assertEqual(training_target(0, [], None), False)
        self.assertEqual(training_target(0, [2], None), False)
        self.assertEqual(training_target(0, [2, 2], None), False)
        self.assertEqual(training_target(0, [30, 1, 2, 2], None), False)
        self.assertEqual(training_target(0, [30, 2, 1, 2], None), False)
        self.assertEqual(training_target(0, [30, 1, 2, 1], None), False)
        self.assertEqual(training_target(0, [30, 3.8, 2.0, 2.9], None), True)
        self.assertEqual(training_target(0, [2, 30000000, 2, 2, 2], None), True)

    def test_epoch_reached(self):
        training_target = targets.epoch_reached(3)

        # Test argument validation.
        with self.assertRaises(ValueError) as context:
            targets.epoch_reached(-1)
        self.assertEqual(
            "specify an epoch greater than or equal to 0",
            str(context.exception))

        # Test processing.
        self.assertEqual(training_target(0, [], None), False)
        self.assertEqual(training_target(2, [], None), False)
        self.assertEqual(training_target(3, [], None), True)
        self.assertEqual(training_target(4, [], None), True)
        self.assertEqual(training_target(4938, [], None), True)

    def test_time_elapsed(self):
        time_elapsed = targets.time_elapsed(1.5)

        # Test argument validation.
        with self.assertRaises(ValueError) as context:
            targets.time_elapsed(-0.00000001)
        self.assertEqual(
            "specify a time period greater than or equal to 0",
            str(context.exception))

        # Test processing.
        self.assertEqual(time_elapsed(None, None, 0), False)
        self.assertEqual(time_elapsed(None, None, 89.9999999), False)
        self.assertEqual(time_elapsed(None, None, 90), True)
        self.assertEqual(time_elapsed(None, None, 90000000), True)

    def test_infinity_reached(self):
        target_reached = targets.infinity_reached()
        self.assertEqual(target_reached(None, None, None), False)
