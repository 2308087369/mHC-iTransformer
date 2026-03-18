# Time Series Forecasting Experiment Report

## 1. Executive Summary

This report presents a comparative study of seven time series forecasting models on seven widely used benchmark datasets. Our primary goal is to assess the behavior of the proposed **MHC_iTransformer** and the newly integrated **DUET** model against strong baselines, including **PatchTST**, **iTransformer**, **TimeFilter**, and **LSTM**.

All experiments were conducted under a unified evaluation pipeline with consistent preprocessing, dataset splits, and metrics. For high-dimensional datasets such as **Traffic** and **Electricity**, **PCA** was applied to reduce the input dimension to 30 in order to avoid CUDA out-of-memory issues and keep training practical on a single GPU.

The main observations are straightforward:
- **PatchTST** remains the strongest general-purpose baseline on several standard benchmarks.
- **TimeFilter** is especially effective on datasets with stronger frequency-domain regularities.
- **MHC_iTransformer** is most competitive on the PCA-reduced high-dimensional datasets.
- **DUET** delivers stable and competitive performance, with particularly strong results on **Traffic**.

## 2. Experiment Setup

### 2.1 Models

The following models were included in the main comparison:
1. **TimeFilter**: A recent forecasting model based on frequency-domain filtering.
2. **iTransformer**: An inverted Transformer architecture for multivariate forecasting.
3. **MHC_iTransformer**: Our modified iTransformer with multi-head channel attention.
4. **PatchTST**: A patch-based Transformer baseline.
5. **LSTM**: A classic recurrent baseline.
6. **DUET**: An integrated Dual-Explorer time series forecasting model.

In addition, we ran a supplementary experiment with **AttnRes_iTransformer**, an iTransformer variant that introduces Attention Residuals. This model was evaluated separately on **Electricity** to study whether the technique is useful for relatively shallow forecasting networks.

### 2.2 Datasets

The experiments cover seven standard benchmark datasets:

| Dataset | Domain | Original Dim | Training Dim | Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **ETTh2** | Electricity Transformer Temperature | 7 | 7 | Hourly |
| **ETTm1** | Electricity Transformer Temperature | 7 | 7 | 15-min |
| **ETTm2** | Electricity Transformer Temperature | 7 | 7 | 15-min |
| **Weather** | Meteorology | 21 | 21 | 10-min |
| **Traffic** | Traffic Flow | 862 | **30 (PCA)** | Hourly |
| **Electricity** | Electricity Load | 321 | **30 (PCA)** | Hourly |
| **Exchange** | Exchange Rate | 8 | 8 | Daily |

### 2.3 Evaluation Metrics

We report four standard regression metrics, where lower values indicate better performance:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **nRMSE**: Normalized RMSE

## 3. Quantitative Results

Table 1 summarizes the test-set performance of all models. Bold values indicate the best result within each dataset and metric.

| Dataset | Model | MAE | MSE | RMSE | nRMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ETTh2** | TimeFilter | 0.3483 | 0.3006 | 0.5483 | 0.0684 |
| | iTransformer | 0.3495 | 0.3013 | 0.5489 | 0.0685 |
| | MHC_iTransformer | 0.3677 | 0.3327 | 0.5768 | 0.0720 |
| | PatchTST | **0.3459** | **0.2962** | **0.5443** | **0.0679** |
| | LSTM | 0.3809 | 0.3535 | 0.5945 | 0.0742 |
| | DUET | 0.3492 | 0.2974 | 0.5454 | 0.0681 |
| **ETTm1** | TimeFilter | 0.3704 | 0.3344 | 0.5783 | 0.0651 |
| | iTransformer | 0.3692 | 0.3352 | 0.5790 | 0.0651 |
| | MHC_iTransformer | 0.3703 | 0.3379 | 0.5813 | 0.0654 |
| | PatchTST | **0.3666** | **0.3336** | **0.5776** | **0.0650** |
| | LSTM | 0.3987 | 0.3758 | 0.6131 | 0.0690 |
| | DUET | 0.3742 | 0.3435 | 0.5860 | 0.0659 |
| **ETTm2** | TimeFilter | **0.2576** | **0.1746** | **0.4178** | **0.0514** |
| | iTransformer | 0.2629 | 0.1816 | 0.4262 | 0.0525 |
| | MHC_iTransformer | 0.2641 | 0.1846 | 0.4297 | 0.0529 |
| | PatchTST | 0.2656 | 0.1788 | 0.4228 | 0.0520 |
| | LSTM | 0.2643 | 0.1850 | 0.4301 | 0.0529 |
| | DUET | 0.2692 | 0.1872 | 0.4326 | 0.0533 |
| **Weather** | TimeFilter | **0.2030** | **0.1576** | **0.3969** | **0.0379** |
| | iTransformer | 0.2118 | 0.1685 | 0.4105 | 0.0391 |
| | MHC_iTransformer | 0.2045 | 0.1598 | 0.3997 | 0.0381 |
| | PatchTST | 0.2110 | 0.1657 | 0.4071 | 0.0388 |
| | LSTM | 0.2169 | 0.1677 | 0.4095 | 0.0391 |
| | DUET | 0.2117 | 0.1606 | 0.4007 | 0.0382 |
| **Traffic** | TimeFilter | 0.5454 | 0.6709 | 0.8191 | 0.0487 |
| | iTransformer | 0.5299 | 0.6492 | 0.8058 | 0.0479 |
| | MHC_iTransformer | 0.5293 | **0.6481** | **0.8051** | **0.0479** |
| | PatchTST | 0.5398 | 0.6718 | 0.8196 | 0.0487 |
| | LSTM | 0.5501 | 0.6573 | 0.8108 | 0.0482 |
| | DUET | **0.5277** | 0.6508 | 0.8067 | 0.0480 |
| **Electricity** | TimeFilter | 0.5569 | 0.5795 | 0.7613 | 0.0629 |
| | iTransformer | 0.5432 | 0.5586 | 0.7474 | 0.0618 |
| | MHC_iTransformer | **0.5421** | **0.5573** | **0.7466** | **0.0617** |
| | PatchTST | 0.5517 | 0.5735 | 0.7573 | 0.0626 |
| | LSTM | 0.6495 | 0.7559 | 0.8695 | 0.0719 |
| | DUET | 0.5498 | 0.5677 | 0.7534 | 0.0623 |
| | AttnRes_iTransformer | 0.5438 | 0.5587 | 0.7475 | 0.0618 |
| **Exchange** | TimeFilter | 0.2074 | 0.0897 | 0.2995 | 0.0443 |
| | iTransformer | 0.2086 | 0.0878 | 0.2963 | 0.0439 |
| | MHC_iTransformer | 0.2105 | 0.0894 | 0.2991 | 0.0443 |
| | PatchTST | **0.2025** | **0.0855** | **0.2924** | **0.0433** |
| | LSTM | 0.2313 | 0.1087 | 0.3297 | 0.0488 |
| | DUET | 0.2067 | 0.0873 | 0.2954 | 0.0437 |

## 4. Result Analysis

### 4.1 Overall Trends

Several clear patterns emerge from the benchmark results:

- **PatchTST** is the strongest all-around baseline, achieving the best overall scores on **ETTh2**, **ETTm1**, and **Exchange**.
- **TimeFilter** performs best on **ETTm2** and **Weather**, suggesting that its frequency-domain inductive bias is well aligned with these datasets.
- **MHC_iTransformer** is most competitive on **Traffic** and **Electricity**, where it achieves the best MSE, RMSE, and nRMSE after PCA-based dimensionality reduction.
- **DUET** is consistently competitive and obtains the best MAE on **Traffic**, indicating strong practical accuracy on a challenging high-dimensional dataset.
- **LSTM**, while still useful as a classical baseline, is generally less competitive than the Transformer-based approaches.

### 4.2 Best Model by Dataset

Using MSE as the primary reference metric, the best-performing model on each dataset is:

| Dataset | Best Model | Observation |
| :--- | :--- | :--- |
| **ETTh2** | PatchTST | Strong and stable performance on standard ETT benchmarks |
| **ETTm1** | PatchTST | Slight but consistent edge over the other Transformer variants |
| **ETTm2** | TimeFilter | Strongest fit for the dataset's spectral characteristics |
| **Weather** | TimeFilter | Best captures local periodicity and smooth variation |
| **Traffic** | MHC_iTransformer | Best error profile after PCA reduction; DUET has the best MAE |
| **Electricity** | MHC_iTransformer | Most effective on PCA-compressed high-dimensional load data |
| **Exchange** | PatchTST | Best overall generalization on the exchange-rate benchmark |

### 4.3 Impact of PCA on Traffic and Electricity

Applying PCA to reduce **Traffic** and **Electricity** to 30 dimensions was a practical necessity for stable training under limited GPU memory. Importantly, dimensionality reduction did not erase the competitiveness of the stronger models. On the contrary, **MHC_iTransformer** and **DUET** remained highly effective in the reduced feature space, and both outperformed several standard baselines on these two datasets.

This result suggests that carefully compressed multivariate inputs can still preserve enough structure for modern forecasting architectures, especially when the model is designed to exploit inter-channel relationships.

### 4.4 Supplementary Analysis of Attention Residuals

We also conducted a depth study on **Electricity** to evaluate **AttnRes_iTransformer**.

- **Observation**: AttnRes_iTransformer performs slightly worse than the baseline iTransformer, with MSE degradation of roughly **0.3% to 2.7%** across the tested depths.
- **Interpretation**: Attention Residuals were originally proposed to stabilize very deep networks such as large language models. In relatively shallow forecasting architectures, the added complexity does not appear to provide a measurable benefit and may instead introduce unnecessary noise.

For this reason, the current evidence does not support replacing standard residual connections with Attention Residuals in this experimental setting.

## 5. Visual Evidence

The figures in the `figures/` directory are consistent with the quantitative results:

1. **Prediction curves** (`{dataset}_prediction.png`)
   These plots show that **PatchTST**, **MHC_iTransformer**, and **DUET** generally track the dominant temporal trends well. On **Traffic**, DUET aligns especially closely with the ground truth, which matches its strong MAE result.

2. **Metric comparison charts** (`comparison_mae.png`, `comparison_mse.png`, `comparison_nrmse.png`)
   The bar charts highlight the broad strength of **PatchTST** on the ETT-style datasets and the advantage of **MHC_iTransformer** on **Traffic** and **Electricity**.

## 6. Conclusion

The experiments support the following conclusions:

1. **PatchTST** is the most reliable default choice across standard benchmark datasets.
2. **TimeFilter** is particularly effective when frequency-domain structure is prominent, as seen on **ETTm2** and **Weather**.
3. **MHC_iTransformer** is most promising on high-dimensional forecasting tasks after PCA compression, especially on **Traffic** and **Electricity**.
4. **DUET** has been integrated successfully and delivers competitive, stable performance, with especially strong practical value on **Traffic**.
5. **AttnRes_iTransformer** does not currently show a clear advantage in shallow time series forecasting settings.

Overall, the benchmark results suggest that no single model dominates every dataset. Model choice should therefore depend on dataset characteristics, computational constraints, and the target evaluation metric.
