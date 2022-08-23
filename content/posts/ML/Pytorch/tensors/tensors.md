---
title: "[PyTorch] Tensors (張量)"
date: 2022-08-17
tags: [PyTorch, Machine Learning, Deep Learning]
categories: [PyTorch, Machine Learning/Deep Learning]
---


# Tensors

> Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

`張量 (Tensor)` 類似向量或矩陣，他是一個 n-維 (n-dimensional) 的資料型態，像是:
- 0 維: scalar
- 1 維: vector
- 2 維: matrix
- n 維: n-dimensional array (ndarray)

他是 Pytorch 最重要的資料結構，也是深度學習裡進行運算的基本元素。


> Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.

Pytorch 的 tensor 在使用與操作上跟 Numpy 的 ndarray 有諸多的相似之處，但重要的是 `Pytorch 的 tensor 可以支援 CUDA 的 GPU 加速`，使我們可以使用 GPU 進行深度學習。

而且 Tensor 跟 Numpy arrays 可以輕鬆的轉換:

    因為他們之間共享相同的底層記憶體，從而無須複製 data

- 詳見: [Bridge with NumPy]()

> Tensors are also optimized for automatic differentiation (we’ll see more about that later in the Autograd section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!

## 初始化張量 (Initializing a Tensor)

Tensors can be initialized in various ways. Take a look at the following examples:

1. ###### Directly from data

    ```python
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    ```

2. ###### From a Numpy array

    ```python
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    ```

3. ###### From another tensor
    - 保留張量屬性，如: shape, data type

        ```python
        x_ones = torch.ones_like(x_data) # retains the properties of x_data
        print(f"Ones Tensor: \n {x_ones} \n")
        ```
        - [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html): 產生與給定張量相同形狀，但元素全為 1 的張量

    - 改變張量 datatype

        ```python
        x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
        print(f"Random Tensor: \n {x_rand} \n")
        ```
        - [torch.rand_like](https://pytorch.org/docs/stable/generated/torch.rand_like.html): 產生與給定張量相同形狀，但元素為 [0,1) 之間均勻分布的隨機數

4. ###### Create a Tensor

    Create a Tensor with empty, random, or constant values

    ```python
    shape = (2,3,)
    empty_tensor = torch.empty(shape)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Empty Tensor: \n {empty_tensor} \n")
    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    ```
    - [torch.empty](https://pytorch.org/docs/stable/generated/torch.empty.html): 只是單純宣告記憶體區塊，並沒有初始化參數，因此裡面的參數可以是任何值

## 張量屬性 (Attributes of a Tensor)

- shape: Tensor 的大小
- dtype: Tensor 裡面元素的資料型別
- device: Tensor 所在的設備，CPU or GPU

    ```python
    tensor = torch.rand(3,4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}") # cpu
    ```

## 張量操作/運算 (Operations on Tensors)

Over 100 tensor operations, including `arithmetic`, `linear algebra`, `matrix manipulation` (transposing, indexing, slicing), `sampling` and more are comprehensively described [here](https://pytorch.org/docs/stable/torch.html).

##### Standard numpy-like indexing and slicing:

可以像 Numpy array 那樣取用部份資料，如下:

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

##### Arithmetic operations (算術運算)

1. matrix multiplication

    ```python
    tensor = torch.tensor([[1, 1], [2, 2]])

    # This computes the matrix multiplication between two tensors.
    # y1, y2, y3 will have the same value
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)
    ```

    output:

    ```
    tensor([[3, 3],
            [6, 6]])
    ```

2. element-wise product

    ```python
    tensor = torch.tensor([[1, 1], [2, 2]])

    # This computes the element-wise product.
    # z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    ```

    output:

    ```
    tensor([[1, 1],
            [4, 4]])
    ```

##### Single-element tensors

If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using `item()`:

```python
tensor = torch.tensor([[1, 1], [2, 2]])
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

output:

```
6 <class 'int'>
```

##### In-place operations

> Operations that store the result into the operand are called in-place.

PyTorch 中大多數的操作都支持 `inplace` 操作，可以直接對 tensor 進行操作而不需要耗另外的記憶體。
方式非常簡單，在操作的符號後面加上 `_`，例如:

- `x.copy_(y)`,
- `x.t_()`

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

## Bridge with NumPy

Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

### Tensor to NumPy array

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

output:

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

##### Changes in the NumPy array reflects in the tensor.

如果是使用 CPU 來操作 Tensor，那麼 tensor 與 numpy array 會共享同一個記憶體，所以操作會被同步，如下:

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

output:

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy array to Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

##### Changes in the NumPy array reflects in the tensor.

如果是使用 CPU 來操作 Tensor，那麼 tensor 與 numpy array 會共享同一個記憶體，所以操作會被同步，如下:

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

output:

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

## 使用 GPU

Pytorch 中的各種張量運算皆可以在 GPU 上運行。

By default, tensors are created on the CPU. We can move tensors to the GPU using [.to()](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to) method.

> Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!


```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```
- [torch.cuda.is_available()](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html): 確認電腦上到底有沒有 GPU

更多 CUDA 相關操作，請看[這邊](https://pytorch.org/docs/stable/notes/cuda.html)

##### 注意: NumPy 只能在 CPU 上操作使用，因此如果需要轉回 NumPy ，需要先把 Tensor 送回 CPU 在操作。

## Reference

- [PyTorch/Tutorials/Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html#)
- [【PyTorch 內部機制】Tensor, Storage, Strides 張量如何存放在記憶體上？](https://weikaiwei.com/pytorch/tensor-storage-strides/)
- [[PyTorch] Getting Start: 從 Tensor 設定開始](https://clay-atlas.com/blog/2019/08/02/pytorch-%E6%95%99%E5%AD%B8%EF%BC%88%E4%B8%80%EF%BC%89-%E5%BE%9E-tensor-%E8%A8%AD%E5%AE%9A%E9%96%8B%E5%A7%8B/)
- [Tensor and Variable](https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/content/2.1.html)
