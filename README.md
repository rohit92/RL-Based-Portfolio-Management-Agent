# RL-Based Portfolio Management Agent

## Overview
This project applies the Actor-Critic policy gradient framework — the same algorithmic family used in model-based RL for autonomous driving — to sequential financial decision-making. The core insight is that portfolio management is a Markov Decision Process: the agent observes market state, takes actions (buy/sell/hold), and receives risk-adjusted rewards, exactly analogous to an autonomous agent observing sensor inputs and receiving driving rewards.

## MDP Formulation
| Component   | Definition |
|-------------|-----------|
| State S     | 20-day OHLCV + technical indicators window + portfolio state (position, unrealized PnL) |
| Action A    | Discrete(3): Hold=0, Buy=1, Sell=2 |
| Reward R    | Log return − transaction cost − volatility penalty |
| Transition  | Next market observation after executing action |
| Episode     | One year of trading data (≈252 steps) |

## Architecture Diagram
```text
Data (OHLCV CSV)
      |
      v
TradingEnv.step()
      |
      v
Observation (182-dim)
      |
      v
PPO MLP Policy (256 x 256)
      |
      v
Action (Hold / Buy / Sell)
      |
      v
Market Execution
      |
      v
Reward = log return - transaction cost - volatility penalty
      |
      v
Rollout Buffer
      |
      v
Policy + Value Update
```

## Why PPO Over DQN
PPO is an Actor-Critic method: it learns both a policy that selects actions and a value function that estimates future return. This is a natural fit for trading because the agent needs a stable policy under noisy observations while still estimating whether a state is attractive.

DQN is value-based and can struggle with action aliasing in trading: very different market contexts can make hold, buy, or sell appear similarly valued in one-step targets, especially when daily rewards are sparse and noisy. PPO optimizes the policy directly, which makes it better aligned with sequential allocation decisions.

PPO's clipped surrogate objective prevents destructive policy updates by bounding how much the new policy can move away from the old one during each optimization phase. That matters in markets because a single unstable update can collapse into degenerate behavior such as always holding or constantly trading.

The autonomous-driving connection is deliberate: both domains use policy-gradient learning under uncertainty, shaped rewards, and safety-aware penalties. Here, "safe" means avoiding unnecessary volatility and transaction churn; in driving, it means avoiding unsafe trajectories.

## Results
Out-of-sample AAPL test period: 2023-01-01 to 2023-12-31.

| Agent        | Annual Return | Sharpe | Max Drawdown | Win Rate | Calmar |
|-------------|--------------:|-------:|-------------:|---------:|-------:|
| PPO          | 6.13%         | 1.013  | 2.44%        | 55.56%   | 2.507  |
| Buy-and-Hold | 35.69%        | 1.647  | 15.05%       | 54.15%   | 2.372  |
| Random       | 12.34%        | 0.902  | 9.03%        | 38.96%   | 1.366  |

## Design Decisions
1. **Log return vs arithmetic return in reward**: log returns are additive through time and represent compounding correctly. PPO optimizes cumulative reward, so an additive reward signal is mathematically cleaner than raw arithmetic returns.

2. **Transaction cost penalty**: without a cost term, the policy can exploit tiny daily noise by flipping actions too often. The explicit cost makes overtrading unattractive and reflects brokerage fees, spread, and slippage.

3. **Volatility penalty**: raw return alone encourages unstable strategies. Penalizing recent realized volatility shapes the policy toward risk-adjusted performance, which is closer to how strategies are evaluated at a quant firm.

4. **2023 as the test set**: the test year is strictly after training and validation. This creates a clean out-of-sample evaluation period with no lookahead from training or model selection.

5. **Train-only normalization statistics**: scaler means and standard deviations are fit only on 2010-2020 training data, then reused unchanged for validation and test. Fitting on later data would leak future distributional information into the model.

6. **Long/flat action space**: the environment intentionally excludes short selling and continuous sizing. This keeps the MDP focused on whether policy-gradient methods can learn entry and exit timing before adding leverage and allocation complexity.

7. **Full-in/full-out execution**: portfolio value updates only while long. That makes baseline comparisons transparent and keeps reward attribution interpretable in interviews.

8. **Two hidden layers of 256 units**: the policy network is large enough to model nonlinear market and portfolio interactions while remaining small enough for stable CPU training on macOS.

9. **Chronological splits**: financial time series are not IID, so random splits would leak regime information. Every split is date-based.

10. **Gymnasium API**: the environment follows the modern Gymnasium reset/step contract, which keeps it compatible with Stable-Baselines3 and common RL tooling.

## Known Limitations
- Single asset (no portfolio diversification)
- Daily granularity only (no intraday)
- No short selling
- No position sizing (always full in or full out)
- AAPL 2023 had specific macro conditions — results may not generalize
- This is a research project, not financial advice

## How to Run
```bash
pip install -r requirements.txt
python data/download_data.py
python train.py --config configs/default.yaml
python evaluate.py
```

## Repository Layout
```text
rl-trading-agent/
├── data/
│   ├── download_data.py
│   └── raw/
├── env/
│   └── trading_env.py
├── agent/
│   ├── ppo_agent.py
│   └── baselines.py
├── utils/
│   ├── metrics.py
│   └── plot.py
├── configs/
│   └── default.yaml
├── train.py
├── evaluate.py
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
```
