## Quantization Comparison Table

| Metric      | Original Sklearn Model    | Quantized Model        |
|-------------|--------------------------|------------------------|
| R² Score    | 0.5758                   | 0.5729                 |
| Model Size  | 414 bytes (0.40 KB)      | 504 bytes (0.49 KB)    |

The R² score of the original scikit-learn Linear Regression model was approximately 0.576 on the test data set.
This suggests that the model represents about 57.6% of the variance of housing prices.
We performed manual quantization of the model parameters down to unsigned 8-bit integer datatypes,
and then made a prediction using the quantized model from the PyTorch implementation; the R² dropped a little to about 0.573.
This did represent a very small ~0.3% drop in R², suggesting a successful and effective quantization process, with very little loss in predictive performance.

Quantization can be useful for lowering the model size and improving deploy engine efficiency while maintaining accuracy;
quantization methods will be useful when deploying ML models in impromptu contexts with limited resources.

