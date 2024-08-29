def build_dataset(name, cfg, is_train=False, **kwargs):
    if name == "camvid":
        from .camvid_dataset import CamvidDataset
        dataset = CamvidDataset(cfg.data_root, is_train=is_train)

    elif name == "portrait2000":
        from .portrait_dataset import PortraitDataset2000
        transform = cfg.train_transform if is_train else cfg.valid_transform
        dataset = PortraitDataset2000(cfg.data_root, transform=transform, is_train=is_train)

    else:
        raise Exception(f'only support camvid and portrait2000, but got {name}')

    return dataset