train_ds = CacheDataset(
    data=train_files,
    transform=train_trans,
    cache_rate=1.0,
    num_workers=8,
    copy_cache=False,
)
val_ds = CacheDataset(
    data=val_files, transform=val_trans, cache_rate=1.0, num_workers=4, copy_cache=False
)
