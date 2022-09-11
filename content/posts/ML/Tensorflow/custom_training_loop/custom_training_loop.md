---
title: "[Tensorflow] Custom Training Loop"
date: 2022-09-01
tags: [Automatic differentiation]
series: [Machine Learning, TensorFlow]
categories: [ML/DL]
---

# Custom Training Loop

在 `tf.keras` 中已經提供很方便的 `training and evaluation loops`, `fit()` 和 `evaluate()`。

但如果我們想要對 training 或 evaluation 進行更 low-level 的控制的話，
我們需要從頭開始寫自己的 `training and evaluation loops`，如:

- 自定義 model 的學習演算法，同時仍然利用 fit() 的便利性。

    ```
    例如: 利用 fit() 來訓練 GAN
    ```

    我們需要用 model subclassing 的方法創建 model，並且實現 `train_step()`方法，在 model.fit() 的期間會一直重複呼叫此方法。詳細請看: [Customizing what happens in fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/)


## Using the GradientTape

Tensorflow 提供了一個很好用的 API: [tf.GradientTape()](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw) 用於`自動微分 (Automatic Differentiation, AD)`，詳細介紹請看 [here](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/gradientTape) 。

我們可以將其分為以下步驟:

- We open a `for` loop that iterates over `epochs`
- For each epoch, we open a `for` loop that iterates over the `dataset`, in `batches`
- For each batch, we open a `GradientTape()` scope
- Inside this scope, we call the model (`forward pass`) and compute the loss
- Outside the scope, we retrieve the `gradients` of the weights of the model with regard to the loss
- Finally, we use the `optimizer to update the weights` of the model based on the gradients

如下所示:

```python
EPOCHS = 2
for epoch in range(EPOCHS):
    # Iterate over the batches of the dataset.
    for batch_idx, (x_train, y_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations
        # run during the forward pass,
        # which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # forward pass
            predicitions = model(x_train, training=True)
            # Compute the loss value for this minibatch.
            loss = loss_fn(y_train, predicitions)

        # backward pass
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights) # model.trainable_variables

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(
            zip(grads, model.trainable_weights))
```

### Low-level handling of metrics

讓我們添加 `metrics` 來監測這一個 training loop。我們可以在這個 loop 內使用 `built-in metrics` 或是 `custom metrics`，流程如下:

- Instantiate the metric at the start of the loop
- Call [metric.update_state()](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#update_state) after each batch
- Call [metric.result()](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#result) when you need to display the current value of the metric
- Call [metric.reset_states()](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#reset_state) when you need to clear the state of the metric

    > typically at the end of an epoch

#### Example:

Let's use this knowledge to compute SparseCategoricalAccuracy on validation data at the end of each epoch:

```python
EPOCHS = 2

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Instantiate the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

for epoch in range(EPOCHS):
    # Iterate over the batches of the dataset.
    for batch_idx, (x_train, y_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # forward pass
            pred = model(x_train, training=True)
            loss = loss_fn(y_train, pred)

        grads = tape.gradient(loss, model.trainable_weights) # model.trainable_variables
        optimizer.apply_gradients(
            zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_train, pred)

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_val, y_val in val_dataset:
        val_pred = model(x_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_val, val_pred)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
```

### Speeding-up your training step with [tf.function](https://www.tensorflow.org/api_docs/python/tf/function)

Tensorflow 2 中默認的 runtime 是 [eager execution.](https://www.tensorflow.org/guide/basics)，這模式適合 debigging ，但是效能比較差，可以透過加上 [@tf.function ](https://www.tensorflow.org/api_docs/python/tf/function) 裝飾器(decorator)將函式編譯成靜態圖 (static graph)。

> Graph compilation has a definite performance advantage.
> 靜態圖會做優化

#### Example

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value
```

你會發現與沒加上 [@tf.function ](https://www.tensorflow.org/api_docs/python/tf/function) 的比起來速度變快了!

### Low-level handling of losses tracked by the model

Layers & models recursively track any losses created during the forward pass by layers that call `self.add_loss(value)`. The resulting list of scalar loss values are available via the property `model.losses` at the end of the forward pass.

If you want to be using these loss components, you should sum them and add them to the main loss in your training step.

#### Example

Consider this layer, that creates an activity regularization loss:

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs
```

Let's build a really simple model that uses it:

```python
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu")(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

Here's what our training step should look like now:

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
        # Add any extra losses created during the forward pass.
        loss += sum(model.losses)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, pred)
    return loss
```
