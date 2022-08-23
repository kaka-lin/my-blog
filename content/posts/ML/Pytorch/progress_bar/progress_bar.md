---
title: "[PyTorch] Various Progress Bar in PyTorch"
date: 2022-08-19
tags: [PyTorch, Machine Learning, Deep Learning, progressbar]
categories: [PyTorch, Machine Learning/Deep Learning]
---

# Pytorch Progress bar

The collection of the `progress bar` methods for `PyTorch` when training model

詳細 code 請看:

- [progress_bar_tqdm.py](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/progress_bar/progress_bar_tqdm.py)
- [progress_bar_keras.py](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/progress_bar/progress_bar_keras.py)
- [progress_bar_click.py](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/progress_bar/progress_bar_click.py)

## 1. `tqdm`

使用 [tqdm](https://tqdm.github.io/) 來顯示 model training 進度, loss and accuracy，如下:

```python
for epoch in range(NUM_EPOCHS):
    with tqdm(train_loader) as pbar:
        for idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # forward
            pred = model(x)
            loss = loss_fn(pred, y)  # calculate loss

            # backward
            optimizer.zero_grad() # clear gradien
            loss.backward()
            optimizer.step()  # update parameters

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / BATCH_SIZE

            pbar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            pbar.set_postfix(loss=loss.item(), acc=accuracy)
```
- `pbar.set_postfix`: 顯示進度情況

結果如下:

```bash
Epoch [1/2]: 100%|█████████████████████████████| 125/125 [00:14<00:00,  8.39it/s, acc=0.125, loss=12]
Epoch [2/2]: 100%|███████████████████████████| 125/125 [00:22<00:00,  5.64it/s, acc=1, loss=0.000991]
```

## 2. `keras progress bar`

[tf.keras.utils.Progbar](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar)

```python
for epoch in range(NUM_EPOCHS):
    n_batches = len(train_loader)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    bar = tf.keras.utils.Progbar(target=n_batches)
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)  # calculate loss

        # backward
        optimizer.zero_grad() # clear gradients
        loss.backward()
        optimizer.step()  # update parameters

        # get the index of the max log-probability
        pred = pred.max(1, keepdim=True)[1]
        correct = pred.eq(y.view_as(pred)).sum().item()
        accuracy = correct / BATCH_SIZE

        bar.update(idx,
            values=[("loss", loss.item()), ("acc", accuracy)])
    print()

```
- `update(current, values=None, finalize=None)`:
  - current: inedex of current step

結果如下:

```bash
Epoch 1/2
124/125 [============================>.] - ETA: 0s - loss: 10.9827 - ass: 0.1020
Epoch 2/2
124/125 [============================>.] - ETA: 0s - loss: 0.1098 - ass: 0.9770
```

## 3. `click`

[click.progressbar](https://click.palletsprojects.com/en/latest/api/#click.progressbar)

```python
for epoch in range(NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
    with click.progressbar(iterable=train_loader,
                           label='',
                           show_percent=True, show_pos=True,
                           item_show_func=metrics_report_func,
                           fill_char='=', empty_char='.',
                           width=36) as bar:
        for idx, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)

            # forward
            pred = model(x)
            loss = loss_fn(pred, y)  # calculate loss

            # backward
            optimizer.zero_grad() # clear gradients
            loss.backward()
            optimizer.step()  # update parameters

            # get the index of the max log-probability
            pred = pred.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / BATCH_SIZE

            bar.current_item = [loss, accuracy]
            final_loss = loss
            final_accuracy = accuracy

        bar.current_item = [final_loss, final_accuracy]
        bar.render_progress()
```
- `item_show_func`: shows the current item, not the previous one

    input 的 (x, y) 也會進我們自己定義的 `metrics_report_func` 裡面，所以:

    1. 更新 `current_item` 為想要顯示的 item ，如: loss, accurancy
    2. 自定義的 func (`metrics_report_func`) 裡面要判斷 dim ，以此來分別現在的是 input 還是我們要的 output 結果，如下所示:

        ```python
        def metrics_report_func(x):
            if x is not None:
                if x[0].dim() == 0:
                    loss, accuracy = x
                    return 'loss: {:.4f} - acc: {:.4f}'.format(loss.item(), accuracy)
        ```

結果如下:

```bash
Epoch 1/2
  [====================================]  125/125  100%  loss: 8.5815 - acc: 0.2500
Epoch 2/2
  [====================================]  125/125  100%  loss: 0.0000 - acc: 1.0000
```
