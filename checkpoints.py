from easydict import EasyDict as edict

class CheckPoints:
    def latest(config):
        return edict({'epoch': 1}), 2

    def save(config, model, optim_state, epoch):
        pass