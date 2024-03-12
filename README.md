# Bitcoin Price Prediction with LSTM

This project leverages Long Short-Term Memory (LSTM) neural networks to predict the price of Bitcoin based on historical price data. By using technical indicators such as the Moving Average Convergence Divergence (MACD) and the Relative Strength Index (RSI) as features, we aim to create a model that can accurately forecast future price movements.

## Features

- **Data Fetching**: Automatically retrieves Bitcoin price data from the CoinGecko API.
- **Technical Indicators**: Calculates MACD and RSI to use as features for the LSTM model.
- **Data Preprocessing**: Includes normalization and splitting of data into training and test sets.
- **LSTM Model**: Utilizes a Sequential model with LSTM layers for making predictions.
- **Evaluation**: Assesses model performance using the Root Mean Squared Error (RMSE) metric.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x installed
- An internet connection for fetching data

## Dependencies

Install the required Python libraries using pip:

```
pip install numpy pandas requests scikit-learn tensorflow ta
```

## Usage

To use this project, follow these steps:

1. **Fetch the Bitcoin price data**:

```python
data = fetch_btc_data()
```

2. **Preprocess the data**: The script automatically handles data normalization and splitting.

3. **Build and train the LSTM model**:

```python
model.fit(X_train, y_train, epochs=25, batch_size=32)
```

4. **Predict future prices and evaluate the model**:

```python
predicted = model.predict(X_test)
# Calculate RMSE
print(f"Root Mean Squared Error: {rmse}")
```

## Contributing to Bitcoin Price Prediction with LSTM

To contribute, follow these steps:
1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Contact

If you want to contact me, you can reach me at `avatsa@stevens.edu`.
