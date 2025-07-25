# Zeru Task Submission: DeFi Wallet Credit Scoring Model 

This project implements a wallet-level credit scoring system using transaction data from the Aave V2 protocol. The objective is to assign a credit score between 0 and 1000 to each wallet based on historical behavior.

## Problem Statement

Using 100K raw transaction records from the Aave V2 protocol, the goal is to engineer meaningful behavioral features from wallet activity and compute a robust credit score that reflects user reliability or risk.

## Repository Contents

- `scores.py`: Python script that processes JSON data and outputs credit scores
- `wallet_scores.csv`: Generated CSV file containing wallet-level features and scores
- `score_distribution.png`: Score distribution bar chart
- `README.md`: Project overview
  
## Engineered Features

The following features are extracted per wallet:

- deposit_volume, borrow_volume, repay_volume, redeem_volume
- deposit_count, borrow_count, repay_count, redeem_count
- liquidation_count
- active_days: Number of days between first and last transaction
- token_diversity: Number of unique tokens interacted with
- total_tx: Sum of all valid transactions

## Scoring Logic

Each wallet receives a base score of 500, which is adjusted based on:

- Positive indicators:
  - Higher deposit and repay volumes
  - High repay-to-borrow ratios
  - Long active duration
  - High token diversity

- Negative indicators:
  - Borrowing without repayment
  - Low transaction volume
  - Liquidation events
  - Inactivity or bot-like behavior

The final score is clipped between 0 and 1000.

## Output

After execution, the following are generated:

- `wallet_scores.csv`: Credit scores and feature breakdown
- `score_distribution.png`: Histogram of score ranges

## How to Run

1. Place the `user-wallet-transactions.json` file inside a `Data/` directory.
2. Run the script:

```bash
python scores.py
