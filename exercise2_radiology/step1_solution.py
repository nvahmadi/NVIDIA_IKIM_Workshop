# Part I: Initialize scaler
scaler = torch.cuda.amp.GradScaler()

# Part II: Scale gradients
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
