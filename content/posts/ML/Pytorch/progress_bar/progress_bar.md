---
title: "[PyTorch] Various Progress Bar in PyTorch"
date: 2022-08-19
tags: [progressbar]
series: [Machine Learning, PyTorch]
categories: [ML/DL]
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
    n_batches = len(train_loader)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    with tqdm(train_loader, total=n_batches,
              bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:36}{r_bar}') as pbar:
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

            # pbar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            pbar.set_postfix({
                'loss': loss.item(),
                'accuracy': accuracy})
```
- `pbar.set_postfix`: 顯示進度情況

結果如下:

```bash
Epoch 1/2
     100%|████████████████████████████████████| 125/125 [00:01<00:00, 96.80it/s, loss=7.79, accuracy=0.375]
Epoch 2/2
     100%|████████████████████████████████████| 125/125 [00:00<00:00, 286.94it/s, loss=0, accuracy=1]
```

## 2. `keras progress bar`

[tf.keras.utils.Progbar](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar)

```python
for epoch in range(NUM_EPOCHS):
    n_batches = len(train_loader)
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    bar = tf.keras.utils.Progbar(target=n_batches,
                                 stateful_metrics=["loss", "accuracy"])
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
            values=[("loss", loss.item()), ("accuracy", accuracy)])
    print()

```
- `update(current, values=None, finalize=None)`:
  - current: inedex of current step
  - values: List of tuples: `(name, value_for_last_step)`.

    ***這邊注意: 如果要顯示每一個 batch 的 loss, accuracy, 需在 `stateful_metrics` 加上相對應的名字

    > If name is in `stateful_metrics`, value_for_last_step will be displayed as-is.
    > Else, an average of the metric over time will be displayed.

結果如下:

```bash
Epoch 1/2
122/125 [============================>.] - ETA: 0s - loss: 11.8544 - accuracy: 0.1250
Epoch 2/2
110/125 [=========================>....] - ETA: 0s - loss: 0.0000e+00 - accuracy: 1.0000
```

## 3. `click`

[click.progressbar](https://click.palletsprojects.com/en/latest/api/#click.progressbar)

```python
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
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
    2. 自定義的 func (`metrics_report_func`) 裡面要判斷 dim ，以此來分別現在的是 input 還是我們要的 output 結果
    3. 用一組變數儲存上一個 batch 的結果，以防止 `metrics_report_func` return `None`。
        > 會跳很快，導致看不到 mtrics

    如下所示:

    ```python
    g_loss, g_accuracy = None, None

    def metrics_report_func(x):
        # using global variables for storing loss and accuracy
        global g_loss
        global g_accuracy

        if x is not None:
            if x[0].dim() == 0:
                loss, accuracy = x
                g_loss, g_accuracy = loss, accuracy # store loss, accuracy into global variables
                return f'loss: {loss.item():.4f} - accuracy: {accuracy:.4f}'
            else:
                if g_loss is not None:
                    return f'loss: {g_loss.item():.4f} - accuracy: {g_accuracy:.4f}'
    ```

結果如下:

```bash
Epoch 1/2
  [====================================]  125/125  100%  loss: 9.6811 - accuracy: 0.1250
Epoch 2/2
  [====================================]  125/125  100%  loss: 0.0000 - accuracy: 1.0000
```
