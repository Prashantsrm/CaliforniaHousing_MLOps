# California Housing MLOps

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Quantization Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 800px;
            margin: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1.5em;
        }
        th, td {
            border: 1px solid #333;
            padding: 8px 12px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
        caption {
            caption-side: top;
            font-weight: bold;
            font-size: 1.2em;
            padding-bottom: 0.5em;
        }
    </style>
</head>
<body>
    <h2>Quantization Comparison Table</h2>

    <table>
        <caption>Model Performance and Size Comparison</caption>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Original Sklearn Model</th>
                <th>Quantized Model</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>R² Score</td>
                <td>0.5758</td>
                <td>0.5729</td>
            </tr>
            <tr>
                <td>Model Size</td>
                <td>414 bytes (0.40 KB)</td>
                <td>504 bytes (0.49 KB)</td>
            </tr>
        </tbody>
    </table>

    <p>
        The R² score of the original scikit-learn Linear Regression model was approximately 0.576 on the test data set.
        This suggests that the model represents about 57.6% of the variance of housing prices.
        We performed manual quantization of the model parameters down to unsigned 8-bit integer datatypes,
        and then made a prediction using the quantized model from the PyTorch implementation;
        the R² dropped a little to about 0.573. This did represent a very small ~0.3% drop in R²,
        suggesting a successful and effective quantization process, with very little loss in predictive performance.
    </p>

    <p>
        Quantization can be useful for lowering the model size and improving deploy engine efficiency while maintaining accuracy;
        quantization methods will be useful when deploying ML models in impromptu contexts with limited resources.
    </p>
</body>
</html>

