import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import math

TOKEN_DECIMALS = {
    "USDC": 6, "USDT": 6, "DAI": 18, "WMATIC": 18,
    "WETH": 18, "WBTC": 8, "AAVE": 18, "TUSD": 18, "GUSD": 2
}

def normalize_amount(amount_str, token_symbol):
    try:
        decimals = TOKEN_DECIMALS.get(token_symbol.upper(), 18)
        return float(amount_str) / (10 ** decimals)
    except:
        return 0.0

def extract_features(transactions):
    features = defaultdict(lambda: {
        "deposit_volume": 0,
        "borrow_volume": 0,
        "repay_volume": 0,
        "redeem_volume": 0,
        "deposit_count": 0,
        "borrow_count": 0,
        "repay_count": 0,
        "redeem_count": 0,
        "liquidation_count": 0,
        "timestamps": [],
        "unique_tokens": set()
    })

    for tx in tqdm(transactions):
        wallet = tx.get("userWallet")
        action = tx.get("action", "").lower()
        timestamp = tx.get("timestamp", 0)
        action_data = tx.get("actionData", {})
        token = action_data.get("assetSymbol", "")
        amount = normalize_amount(action_data.get("amount", 0), token)

        if not wallet:
            continue

        f = features[wallet]
        f["timestamps"].append(timestamp)
        if token:
            f["unique_tokens"].add(token)

        if action == "deposit":
            f["deposit_volume"] += amount
            f["deposit_count"] += 1
        elif action == "borrow":
            f["borrow_volume"] += amount
            f["borrow_count"] += 1
        elif action == "repay":
            f["repay_volume"] += amount
            f["repay_count"] += 1
        elif action == "redeemunderlying":
            f["redeem_volume"] += amount
            f["redeem_count"] += 1
        elif action == "liquidationcall":
            f["liquidation_count"] += 1

    rows = []
    for wallet, f in features.items():
        duration_days = (max(f["timestamps"]) - min(f["timestamps"])) / (60 * 60 * 24) if f["timestamps"] else 0
        rows.append({
            "wallet": wallet,
            "deposit_volume": round(f["deposit_volume"], 6),
            "borrow_volume": round(f["borrow_volume"], 6),
            "repay_volume": round(f["repay_volume"], 6),
            "redeem_volume": round(f["redeem_volume"], 6),
            "deposit_count": f["deposit_count"],
            "borrow_count": f["borrow_count"],
            "repay_count": f["repay_count"],
            "redeem_count": f["redeem_count"],
            "liquidation_count": f["liquidation_count"],
            "active_days": round(duration_days, 2),
            "token_diversity": len(f["unique_tokens"]),
            "total_tx": (
                f["deposit_count"] + f["borrow_count"] +
                f["repay_count"] + f["redeem_count"]
            )
        })

    return pd.DataFrame(rows)

def compute_scores(df):
    scores = []
    for _, row in df.iterrows():
        score = 500

        if row["deposit_volume"] < 10:
            score -= 100
        if row["borrow_volume"] > 0 and row["repay_volume"] == 0:
            score -= 200
        if row["liquidation_count"] > 0:
            score -= row["liquidation_count"] * 100
        if row["repay_volume"] > 0:
            score += math.log(row["repay_volume"] + 1) * 20
        if row["deposit_volume"] > 0:
            score += math.log(row["deposit_volume"] + 1) * 10
        if row["active_days"] > 0:
            score += min(row["active_days"] / 30, 12) * 5
        if row["token_diversity"] > 0:
            score += row["token_diversity"] * 5
        if row["borrow_count"] > 0:
            repay_ratio = row["repay_count"] / row["borrow_count"]
            score += repay_ratio * 100
        if row["total_tx"] < 3:
            score -= 100

        score = min(1000, max(0, score))
        scores.append(score)

    df["credit_score"] = scores
    return df

def plot_distribution(df, output_path="score_distribution.png"):
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1100, 100)
    df["score_range"] = pd.cut(df["credit_score"], bins=bins)
    score_counts = df["score_range"].value_counts().sort_index()
    sns.barplot(x=score_counts.index.astype(str), y=score_counts.values)
    plt.xticks(rotation=45)
    plt.xlabel("Credit Score Range")
    plt.ylabel("Wallet Count")
    plt.title("Wallet Credit Score Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_json(json_path="Data/user-wallet-transactions.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = extract_features(data)
    df = compute_scores(df)
    plot_distribution(df, "score_distribution.png")
    df.to_csv("wallet_scores.csv", index=False)
    print("Done. Saved: wallet_scores.csv and score_distribution.png")
    return df

if __name__ == "__main__":
    process_json()
