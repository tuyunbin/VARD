def create_dataset(cfg, split='train'):
    dataset = None
    data_loader = None
    if cfg.data.dataset == 'rcc_dataset':
        from datasets.rcc_dataset import RCCDataset, RCCDataLoader

        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == 'rcc_dataset_spot':
        from datasets.rcc_dataset_spot import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)

    elif cfg.data.dataset == 'rcc_dataset_edit':
        from datasets.rcc_dataset_edit import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)

    elif cfg.data.dataset == 'rcc_dataset_transformer':
        from datasets.rcc_dataset_transformer import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == 'rcc_dataset_spot_transformer':
        from datasets.rcc_dataset_spot_transformer import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == 'rcc_dataset_edit_transformer':
        from datasets.rcc_dataset_edit_transformer import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    else:
        raise Exception('Unknown dataset: %s' % cfg.data.dataset)
    
    return dataset, data_loader
