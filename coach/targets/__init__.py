def loss_stagnation(epsilon=1, conservativity=3):
    if conservativity < 2:
        raise ValueError("specify a conservativity greater than or equal to 2")

    def target_function(epoch_number, losses, started_at):
        if len(losses) < conservativity:
            return False

        for loss in losses[-conservativity:-1]:
            difference = abs(losses[-1] - loss)
            if difference >= epsilon:
                return False

        return True

    return target_function

def epoch_reached(epoch):
    if epoch < 0:
        raise ValueError("specify an epoch greater than or equal to 0")

    def target_function(epoch_number, losses, started_at):
        return epoch_number >= epoch

    return target_function

def time_elapsed(time_period): # `time_period` in minutes!
    if time_period < 0:
        raise ValueError("specify a time period greater than or equal to 0")

    def target_function(epoch_number, losses, started_at):
        time_period_in_seconds = time_period * 60
        return started_at >= time_period_in_seconds

    return target_function

def infinity_reached():
    def target_function(epoch_number, losses, started_at):
        return False

    return target_function
