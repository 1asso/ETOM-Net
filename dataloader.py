class DataLoader:
    def create(opt):
        loaders = []
        # import data.datasets.TOMDataset

        # for i, split in enumerate(['train', 'val']):
        #     dataset = TOMDataset(opt, split)
        #     loaders[i] = DataLoader(dataset, opt, split)
        return 1, 2

    def __init__(dataset, opt, split):
        manual_seed = opt.manual_seed
        

