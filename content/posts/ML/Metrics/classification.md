---
title: "[Classification] 分類指標 (Classification Metrics)"
date: 2022-08-15
tags: [Classification, Metrics]
series: [Machine Learning]
categories: [ML/DL]
---

# 分類指標 (Classification Metrics)

## Confuion Matrix (混淆矩陣)

|      |  Predict Positive  |  Predict Negative   |
| :--: | :--------: | :---------: |
|  Actual Positive  | True Positive (TP)  | False Negative (FN) |
|  Actual Negative  | False Positive (FP) | True Negative (TN)  |


## Recall and Precision

### Recall (召回率)

```
Recall = TP / (TP + FN)
```

表示在所有正樣本中，能夠預測到多少個正樣本。 (`針對原來的樣本`)

### Precision (準確率)

```
Precision = TP / (TP + FP)
```

為在所有預測為正樣本中，有多少實際為正樣本。 (`針對預測結果`)

### 應用場景

#### 1. 當今天你加裝了一個人臉辨識的門鎖時，哪個指標比較重要呢？

```
準確率比較重要，因為你不會希望別人的臉可以打開你家的鎖，判斷成正樣本就一定要是對的

召回率低的話就只是常常無法判斷出你的臉，無法開門而已
```

#### 2. 廣告投放系統

```
Recall比較重要，因為重視潛在客戶，我全都要 (實際正向客戶中我預測到多少個)

準確率就沒那麼重要，不在意預測正向（廣告投出）答對多少
```

### 小結

1. Precision高，Recall低的模型

    `謹慎的模型`, 雖然常常無法抓到，但只要有預測到的幾乎都是正確的

2. Recall高，Precision低的模型

    `寬鬆的模型`, 雖然有可能預測錯誤，但只要是該預測的都可以預測到

但魚與熊掌不可兼得，如果我們同時想要兼顧兩個指標怎辦呢？這時候就要看 `F1-score`了。

## F1-score (F1-Mesure)

他是`F-score`的一個特例，當`beta=1`時就是`F1-score`。

`F1-score`最理想的數值是`趨近於1`，就是讓precision和recall都有很高的值。

```
F1-score = 2 * ((Precision * Recall) / (Precision + Recall))
```

$$ F1 = 2 * \frac{Precision * Recall}{Precision + Recall}$$


假設兩者皆為1，則`F1-score = 1 (100%)`，代表該演算法有著最佳的精確度

### F-score (F-Mesure)

```
F-score = ((1 + beta)^2 * Precision * Recall / (beta^2 * Precision) + Recall)
```

$$ F = \frac{(1 + \beta^2) * Precision * Recall}{(\beta^2 * Precision) + Recall}$$


1. `beta=0`: 就是Precision
3. `beta=1`: 就是F1-score
2. `beta無限大`: 就是Recall


所以當我想多看重一點Precision時，beta就可以選小一點，當我想多看重Recall時，beta就可以選大一點。
