---
title: "[Tensorflow] Progress bar of custom training loop"
date: 2022-09-02
tags: [Automatic differentiation, progressbar]
series: [Machine Learning, TensorFlow]
categories: [ML/DL]
---

# Progress bar of Tensorflow 2's custom training loop

The collection of the `progress bar` methods for [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw) when training model

詳細 code 請看:

- [progress_bar_tqdm.py](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/custom_training_loop/progress_bar/progress_bar_tqdm.py)
- [progress_bar_keras.py](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/custom_training_loop/progress_bar/progress_bar_keras.py)
- [progress_bar_click.py](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/custom_training_loop/progress_bar/progress_bar_click.py)

## 1. `tqdm`

使用 [tqdm](https://tqdm.github.io/) 來顯示 model training 進度, loss and accuracy，如下:

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

for epoch in range(NUM_EPOCHS):
    n_batches = x_train.shape[0] / BATCH_SIZE
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    with tqdm(train_dataset, total=n_batches,
              bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:36}{r_bar}') as pbar:
        for idx, (x, y) in enumerate(pbar):
            with tf.GradientTape() as tape:
                # forward
                pred = model(x, training=True)
                loss = loss_fn(y, pred)

            # backward
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update training metric after batch
            train_loss.update_state(loss)
            train_accuracy.update_state(y, pred)

            #pbar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            pbar.set_postfix({
                'loss': train_loss.result().numpy(),
                'accuracy': train_accuracy.result().numpy()})

    train_loss.reset_states()
    train_accuracy.reset_states()
```
- `pbar.set_postfix`: 顯示進度情況

結果如下:

```bash
Epoch 1/2
     100%|████████████████████████████████████| 1875/1875.0 [00:14<00:00, 126.51it/s, loss=0.142, accuracy=0.956]
Epoch 2/2
     100%|████████████████████████████████████| 1875/1875.0 [00:14<00:00, 133.67it/s, loss=0.0439, accuracy=0.986]
```

## 2. `keras progress bar`

[tf.keras.utils.Progbar](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar)

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

for epoch in range(NUM_EPOCHS):
    n_batches = x_train.shape[0] / BATCH_SIZE
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    bar = tf.keras.utils.Progbar(target=n_batches,
                                 stateful_metrics=["loss", "accuracy"])
    for idx, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # forward
            pred = model(x, training=True)
            loss = loss_fn(y, pred)
        # backward
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update training metric after batch
        train_loss.update_state(loss)
        train_accuracy.update_state(y, pred)

        bar.update(idx+1,
            values=[("loss", train_loss.result()), ("accuracy", train_accuracy.result())])

    train_loss.reset_states()
    train_accuracy.reset_states()
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
1875/1875 [==============================] - 14s 7ms/step - loss: 0.1366 - accuracy: 0.9585
Epoch 2/2
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0428 - accuracy: 0.9868
```

## 3. `click`

[click.progressbar](https://click.palletsprojects.com/en/latest/api/#click.progressbar)

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

for epoch in range(NUM_EPOCHS):
    n_batches = x_train.shape[0] / BATCH_SIZE
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    with click.progressbar(iterable=train_dataset,
                           label='',
                           show_percent=True, show_pos=True,
                           item_show_func=metrics_report_func,
                           fill_char='=', empty_char='.',
                           width=36) as bar:
        for idx, (x, y) in enumerate(bar):
            with tf.GradientTape() as tape:
                # forward
                pred = model(x, training=True)
                loss = loss_fn(y, pred)

            # backward
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update training metric after batch
            train_loss.update_state(loss)
            train_accuracy.update_state(y, pred)

            # print(train_loss.result().numpy(), train_accuracy.result().numpy())
            bar.current_item = [train_loss.result(), train_accuracy.result()]
            final_loss = train_loss.result()
            final_accuracy = train_accuracy.result()

        bar.current_item = [final_loss, final_accuracy]
        bar.render_progress()

    train_loss.reset_states()
    train_accuracy.reset_states()
```
- `item_show_func`: shows the current item, not the previous one

    input 的 (x, y) 也會進我們自己定義的 `metrics_report_func` 裡面，所以:

    1. 更新 `current_item` 為想要顯示的 item ，如: loss, accurancy
    2. 自定義的 func (`metrics_report_func`) 裡面要判斷 dim ，以此來分別現在的是 input 還是我們要的 output 結果
    3. 用一組變數儲存上一個 batch 的結果，以防止 `metrics_report_func` return `None`。
        > 會跳很快，導致看不到 mtrics

    如下所示:

    ```python
    # declare global variables for storing previous loss, previous accuracy
    g_loss, g_accuracy = None, None

    def metrics_report_func(x):
        # using global variables for storing loss and accuracy
        global g_loss
        global g_accuracy

        if x is not None:
            if tf.size(x[0]).numpy() == 1:
                loss, accuracy = x
                g_loss, g_accuracy = loss, accuracy # store loss, accuracy into global variables
                return f'loss: {loss.numpy():.4f} - accuracy: {accuracy.numpy():.4f}'
            else:
                if g_loss is not None:
                    return f'loss: {g_loss.numpy():.4f} - accuracy: {g_accuracy.numpy():.4f}'
    ```

結果如下:

```bash
Epoch 1/2
  [====================================]  1875/1875  100%  loss: 0.1367 - accuracy: 0.9586
Epoch 2/2
  [====================================]  1875/1875  100%  loss: 0.0448 - accuracy: 0.9860
```
