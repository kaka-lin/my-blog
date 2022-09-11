---
title: "[Tensorflow] Introduction to tf.GradientTape and automatic differentiation"
date: 2022-08-31
tags: [Automatic differentiation]
series: [Machine Learning, TensorFlow]
categories: [ML/DL]
---

# GradientTape

在介紹 [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw) 前我先來看看什麼是`自動微分 (Automatic differentiation, AD)`

## Automatic Differentiation

為了`自動微分(Automatic differentiation)`，TensorFlow 需要:

1. `前向傳播(forward pass)`: 記住以什麼順序發生什麼樣的操作。
2. `反向傳播(backward pass)`: 以相反的順序遍歷這個操作列表來計算梯度。

在 TensorFlow 2 提供 [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw)
用於自動微分，也就是計算某些輸入的梯度 (Gradient)。

Tensorflow 會將在 `tf.GradientTape` 上下文中執行的相關操作記錄到`"磁帶(tape)"`上。
然後 tape 會計算反向傳播中的梯度。

> TensorFlow "records" relevant operations executed inside the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape to compute the gradients of a "recorded" computation using [reverse mode differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

## GradientTape

[tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw) 默認將`所有可訓練的變量(tf.Variable, where trainable=True)`視為`需要監控的 node (watch_accessed_variables=True)`。API 如下:

```python
tf.GradientTape(
    persistent=False,
    watch_accessed_variables=True
)
```

> Record operations for automatic differentiation.

   - `persistent`: Boolean control whether a persistent gradient tape is created.
   - `watch_accessed_variables`: Boolean control whether the tape will automatically `watch` any (trainable) variables accessed while the tape is active.

### Computing gradients

用 [tf.GradientTape.gradient](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw#gradient) 來計算梯度，API 如下:

```python
tf.GradientTape.gradient(
    target,
    sources,
    output_gradients=None,
    unconnected_gradients=tf.UnconnectedGradients.NONE
)
```

> Computes the gradient using operations recorded in context of this tape.
>
   - `target`: Tensor (or list of tensors) to be differentiated.
   - `sources`: a list or nested structure of Tensors or Variables. `target` will be differentiated against elements in `sources`.
   - `output_gradients`: a list of gradients, one for each element of target. Defaults to None.
   - `unconnected_gradients`: a value which can either hold 'none' or 'zero' and alters the value which will be returned if the target and sources are unconnected. The possible values and effects are detailed in 'UnconnectedGradients' and it defaults to 'none'.

#### Example

For example, consider the function `y = x * x`. The gradient at `x = 3.0` can be computed as:

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
dy_dx.numpy()
```

Output:

```sh
6.0
```

### Controlling what the tape watches

The default behavior is to record all operations after accessing a trainable `tf.Variable`. The reasons for this are:

- The tape needs to know which operations to record in the forward pass to calculate the gradients in the backwards pass.
- The tape holds references to intermediate outputs, so you don't want to record unnecessary operations.
- The most common use case involves calculating the gradient of a loss with respect to all a model's trainable variables.

Tape 默認的監控變數只有 `tf.Variable 且 trainable=True`，
其他變數則會計算梯度失敗，如下:

- `tf.Tensor`: not "watched"
- `tf.Varaiable, trainble=False`

對於以上不可訓練或沒有被監控的變量，可以使用 [tf.GradientTape.watch](https://www.tensorflow.org/api_docs/python/tf/GradientTape?hl=zh-tw#watch) 對其進行監控，API 如下:

```python
tf.GradientTape.watch(tensor)
```

> Ensures that tensor is being traced by this tape.

  - `tensor`: a Tensor or list of Tensors.

#### Example

```python
# A trainable variable
x0 = tf.Variable(3.0, name='x0') # tf.Variable

# Not trainable: `trainable=False`
x1 = tf.Variable(3.0, name='x1', trainable=False) # tf.Variable

# Not a variable: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.0 # tf.Tensor

# Not a variable
x3 = tf.constant(3.0, name='x3') # tf.Tensor

with tf.GradientTape() as tape:
    y = (x0**2) + (x1**2) + (x2**2) + (x3**2)

grad = tape.gradient(y, [x0, x1, x2, x3])
for g in grad:
    print(g)
```

Output:

```
tf.Tensor(6.0, shape=(), dtype=float32)
None
None
None
```

To record gradients with respect to a `tf.Tensor`, you need to call `GradientTape.watch(x)`:

```python
x0 = tf.Variable(3.0, name='x0')
x1 = tf.Variable(3.0, name='x1', trainable=False)
x2 = tf.Variable(2.0, name='x2') + 1.0
x3 = tf.constant(3.0, name='x3')
with tf.GradientTape() as tape:
    tape.watch([x1, x2, x3])
    y = (x0**2) + (x1**2) + (x2**2) + (x3**2)

grad = tape.gradient(y, [x0, x1, x2, x3])
for g in grad:
    print(g)
```

Output:

```
tf.Tensor(6.0, shape=(), dtype=float32)
tf.Tensor(6.0, shape=(), dtype=float32)
tf.Tensor(6.0, shape=(), dtype=float32)
tf.Tensor(6.0, shape=(), dtype=float32)
```

#### Disable automatic tracking

By default, GradientTape will automatically watch any trainable variables that are accessed inside the context.

If you want `fine-grained control` over which variables are watched you disable automatic tracking by passing `watch_accessed_variables=False` to the tape constructor.

#### Example

```python
variable_a = tf.Variable(3.0, name='x1')
variable_b = tf.Variable(2.0, name='x2')

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as disable_tracking_tape:
    disable_tracking_tape.watch(variable_a)
    y = variable_a ** 2 # Gradients will be available for `variable_a`.
    z = variable_b ** 3 # No gradients will be available since `variable_b` is
                        # not being watched.
gradient_1 = disable_tracking_tape.gradient(y, variable_a) # 6.0
gradient_2 = disable_tracking_tape.gradient(z, variable_b) # None

print(gradient_1)
print(gradient_2)
```

Output:

```
tf.Tensor(6.0, shape=(), dtype=float32)
None
```

### Compute multiple gradient

By default, the resources held by a `GradientTape` are released as soon as `GradientTape.gradient()` method is called.

To compute multiple gradients over the same computation, create `a persistent gradient tape`. This allows multiple calls to the gradient() method as resources are released when the tape object is garbage collection.

#### Example

```python
x = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as persistent_tape:
    persistent_tape.watch(x)
    y = x * x
    z = y * y
dz_dx = persistent_tape.gradient(z, x) # 108.0 (4*x^3 at x = 3)
dy_dx = persistent_tape.gradient(y, x) # 6.0
print("First derivative of function y = x ^ 4 at x = 3 is", dz_dx.numpy())

# Drop the reference to the tape
del persistent_tape
#persistent_tape # NameError: name 'persistent_tape' is not defined
```

Output:

```
First derivative of function y = x ^ 4 at x = 3 is 108.0
```

### Nested Gradient

GradientTapes can be nested to compute higher-order derivatives.

#### Example

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y = x * x
    dy_dx = tape2.gradient(y, x)
d2y_d2x = tape.gradient(dy_dx, x)

print("Function: y = x * x, x = 3.0")
print("First Derivative:", dy_dx.numpy())
print("Second Derivative:", d2y_d2x.numpy())
```

Output:

```sh
Function: y = x * x, x = 3.0
First Derivative: 6.0
Second Derivative: 2.0
```
