import datetime
import time

from torch_state_control import StateManager
from .aux import minutes_since, seconds_since
from .checkpoint_creator import CheckpointCreator


class Coach:

    def __init__(self, network):
        self.network = network

    def __log__(self, message='', end='\n'):
        print(message, end=end)

    def __log_epoch_summary__(self, epoch, loss, seconds_elapsed):
        minutes_elapsed = seconds_elapsed / 60

        self.__log__(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss:02.15f} "
            f"| Time elapsed: {minutes_elapsed:4.1f} minutes "
            '                                                   '
        )

    def train(self, target, train_one_epoch, measure_performance, checkpoint_frequency, checkpoint=None, settings_notes=None, record=True):
        checkpoint_creator = CheckpointCreator(
            network=self.network,
            measure_performance=measure_performance,
            settings_notes=settings_notes,
            record=record,
            logger=self.__log__)

        checkpoint_creator.load_checkpoint(checkpoint)

        training_started_at = time.time()
        last_checkpoint_at = datetime.datetime.now()
        seconds_elapsed = 0
        epoch = 0
        epochs_since_last_checkpoint = 0

        losses = []
        losses_since_last_checkpoint = []
        while not target(epoch, losses, seconds_elapsed):
            epoch += 1
            started_at = time.time()

            self.__log__(f"Epoch {epoch:4d} | ", end="\r")

            loss = train_one_epoch()
            epochs_since_last_checkpoint += 1

            ended_at = time.time()
            duration = ended_at - started_at

            self.__log__(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss:.10f}, "
                f"Duration: {duration:.2f} s",
                end="\r"
            )

            losses.append(loss)
            losses_since_last_checkpoint.append(loss)

            minutes_since_last_checkpoint = minutes_since(last_checkpoint_at)
            seconds_elapsed = seconds_since(training_started_at)

            if minutes_since_last_checkpoint >= checkpoint_frequency:
                self.__log_epoch_summary__(epoch, loss, seconds_elapsed)
                last_checkpoint_at = checkpoint_creator.create_checkpoint(
                    epochs_since_last_checkpoint,
                    losses_since_last_checkpoint)
                losses_since_last_checkpoint = []
                epochs_since_last_checkpoint = 0

        self.__log_epoch_summary__(epoch, loss, seconds_elapsed)
        checkpoint_creator.create_checkpoint(
            epochs_since_last_checkpoint,
            losses_since_last_checkpoint)
        self.__log__("Training target reached - training ends")
