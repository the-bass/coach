import datetime
import time

from torch_state_control import StateManager
from .aux import minutes_since, seconds_since


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

    def __load_checkpoint__(self, checkpoint):
        if checkpoint == None:
            return

        if checkpoint == -1:
            self.network.load_latest_checkpoint()
        else:
            self.network.load_checkpoint(checkpoint)

        # Log information about the loaded checkpoint.
        loaded_checkpoint = self.network.latest_checkpoint()
        if not loaded_checkpoint:
            return

        self.__log__(
            f"CHECKPOINT RESTORED | "
            f"Checkpoint ID: {loaded_checkpoint.id}, "
            f"Success on TRAIN set: {loaded_checkpoint.train_set_performance}, "
            f"Success on DEV set: {loaded_checkpoint.dev_set_performance} | "
            f"{loaded_checkpoint.created_at:%H:%M:%S, %d.%m.%Y}"
        )

    def __create_checkpoint__(self, measure_performance, notes, epochs_since_last_checkpoint, losses_since_last_checkpoint, record):
        self.__log__(
            'CHECKPOINT | '
            f"Measuring performance on train and dev sets ...",
            end="\r"
        )

        train_set_performance, dev_set_performance = measure_performance()

        if record:
            self.network.save_checkpoint(
                notes=notes,
                train_set_performance=train_set_performance,
                dev_set_performance=dev_set_performance,
                losses_since_last_checkpoint=losses_since_last_checkpoint
            )

        created_at = datetime.datetime.now()

        self.__log__(
            'CHECKPOINT | '
            f"Success on TRAIN set: {train_set_performance}, "
            f"Success on DEV set: {dev_set_performance} | "
            f"{created_at:%H:%M:%S, %d.%m.%Y}"
        )

        return created_at

    def train(self, target, train_one_epoch, measure_performance, checkpoint_frequency, checkpoint=None, notes=None, record=True):
        self.__load_checkpoint__(checkpoint)

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
                last_checkpoint_at = self.__create_checkpoint__(measure_performance, notes, epochs_since_last_checkpoint, losses_since_last_checkpoint, record)
                losses_since_last_checkpoint = []
                epochs_since_last_checkpoint = 0

        self.__log_epoch_summary__(epoch, loss, seconds_elapsed)
        self.__create_checkpoint__(measure_performance, notes, epochs_since_last_checkpoint, losses_since_last_checkpoint, record)
        self.__log__("Training target reached - training ends")
