import time

class StrUtils:
    def build_loss_string(losses, no_total):
        total_loss = 0
        s = ''
        for k, v in losses.items():
            s += '{}: {}, '.format(k, v)
            total_loss += v
        if not no_total:
            s += ' [Total Loss: {}]'.format(total_loss) 
        return s

    def time_left(start_time, n_epochs, batches, cur_epoch, cur_iter):
        cur_time = time.time()
        time_so_far = (cur_time - start_time) / 3600.0
        total_step = n_epochs * batches + cur_iter
        cur_step = (cur_epoch - 1) * batches + cur_iter
        time_left = time_so_far * (total_step / cur_step - 1)
        s = 'Time elapsed: {} hours | Time left: {} hours'.format(time_so_far, time_left)
        return s

    def build_time_string(times, no_total):
        total_time = 0
        s = ''
        for k, v in times.items():
            s += '{}: {} seconds, '.format(k, v)
            total_time += v
        if not no_total:
            s += ' [Total time: {}'.format(total_time)
        return s