# Part I: train transforms
train_transforms.append(
            ToDeviced(keys=["image", "label"], device="cuda:0")
        )

# Part II: val transforms
val_transforms.append(
            ToDeviced(keys=["image", "label"], device="cuda:0")
        )
