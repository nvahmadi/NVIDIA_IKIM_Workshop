train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
