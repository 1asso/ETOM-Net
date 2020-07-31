class DataLoader:
    def create(config):
        loaders = []
        import data.datasets.TOMDataset

        for i, split in enumerate(['train', 'val']):
            dataset = TOMDataset(config, split)
            loaders[i] = DataLoader(dataset, config, split)

    def __init__(dataset, config, split):
        manual_seed = config.manual_seed
        
