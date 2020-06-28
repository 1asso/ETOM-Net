class Trainer:
    def __init__(self, model, config, optim_state):
        print('Initializing Trainer')
        self.config = config
        self.model = model
        self. optim_state = "todo"
        ## TODO

    def test(self, epoch, dataloader, split, *predictor):
        pass

    def train(self, epoch, dataloader, split, *predictor):
        pass
