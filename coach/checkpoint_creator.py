import datetime


class CheckpointCreator:

    def __init__(self, network, measure_performance, settings_notes, record, logger):
        self.network = network
        self.measure_performance = measure_performance
        self.settings_notes = settings_notes
        self.logger = logger
        self.record = record

    def __log__(self, *args, **kwargs):
        if not self.logger:
            return

        self.logger(*args, **kwargs)

    def create_checkpoint(self, epochs_since_last_checkpoint, losses_since_last_checkpoint):
        self.__log__(
            'CHECKPOINT | '
            f"Measuring performance on train and dev sets ...",
            end="\r"
        )

        train_set_performance, dev_set_performance = self.measure_performance()

        if self.record:
            notes = {
                'train_set_performance': train_set_performance,
                'dev_set_performance': dev_set_performance,
                **self.settings_notes
            }

            self.network.save_checkpoint(notes=notes)

        created_at = datetime.datetime.now()

        self.__log__(
            'CHECKPOINT | '
            f"Success on TRAIN set: {train_set_performance}, "
            f"Success on DEV set: {dev_set_performance} | "
            f"{created_at:%H:%M:%S, %d.%m.%Y}"
        )

        return created_at

    def load_checkpoint(self, checkpoint):
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
            f"Checkpoint ID: {loaded_checkpoint.id} | "
            f"Notes: {loaded_checkpoint.notes} | "
            f"{loaded_checkpoint.created_at:%H:%M:%S, %d.%m.%Y}"
        )
