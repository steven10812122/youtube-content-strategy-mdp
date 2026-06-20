# YouTube Content Strategy under Uncertainty: A Markov Decision Process Approach

This project models post-viral YouTube content strategy as a **data-driven Markov Decision Process (MDP)**. It uses 200 long-form videos from five Taiwanese YouTube channels to estimate state transitions, then solves both an unconstrained discounted MDP and a constrained MDP.

The project is designed as an Operations Research / stochastic modeling side project. It is not a neural-network virality predictor. The goal is to show how a dynamic content-performance problem can be formulated with states, actions, transition probabilities, rewards, constraints, and optimal policies.

## Research Question

When a video reaches a high-performance state, what next-video strategy should a creator choose under uncertainty?

More specifically:

1. How does video performance transition across states such as Low, Normal, Hit, and Viral?
2. Can next-video strategy and release timing be modeled as MDP actions?
3. What policy is recommended by value iteration and policy iteration?
4. How does the recommendation change when resource and risk constraints are imposed?

## Dataset

| Item | Value |
|---|---:|
| Creators | 5 |
| Videos | 200 |
| Videos per creator | 40 |
| Valid within-creator transitions | 195 |
| Analysis date | 2026-06-20 |

The unit of analysis is one long-form YouTube video. Shorts and non-standard videos are excluded.

## State Definition

Each video is mapped to a relative performance state using creator-normalized views per day.

$$
r_i = \frac{\text{views per day of video } i}{\text{median views per day of the same creator}}
$$

| Ratio | State | Score |
|---:|---|---:|
| $r_i < 0.7$ | Low | 0 |
| $0.7 \le r_i < 2$ | Normal | 1 |
| $2 \le r_i < 5$ | Hit | 2 |
| $r_i \ge 5$ | Viral | 3 |

This normalization avoids comparing large and small channels by raw views.

## Action Definition

The decision is the next-video strategy and publication deadline:

$$
a = (\text{content strategy}, D_{\max})
$$

where $D_{\max} \in \{7,14,30\}$ days.

| Strategy | Meaning |
|---|---|
| Travel/Experience | Travel, challenge, experiment, experience-style videos |
| Talk/Topic | Talk, interviews, social topics, commentary |
| Review/Tech | Reviews, unboxing, technology, ranking-style content |
| Lifestyle/Other | Lifestyle, daily, miscellaneous content |

## Empirical Markov Chain

The baseline Markov chain estimates:

$$
P(X_{t+1}=j \mid X_t=i)
$$

The empirical transition probability is:

$$
\hat P_{ij} = \frac{N_{ij}}{\sum_j N_{ij}}
$$

where $N_{ij}$ is the number of observed transitions from state $i$ to state $j$.

## Action-Conditioned Transition Model

For the MDP, transition probabilities are estimated conditional on state and action:

$$
P(X_{t+1}=j \mid X_t=s, A_t=a)
$$

Because the dataset is small, the project uses empirical-Bayes smoothing:

$$
\hat P(j \mid s,a) = \frac{N_{saj} + \kappa \hat P(j \mid s)}{N_{sa} + \kappa}
$$

where $\kappa=4$ is the smoothing strength.

## Standard Discounted MDP

The unconstrained MDP maximizes discounted expected net performance:

$$
V(s)=\max_a \left[ R(s,a)+\gamma \sum_{s'} P(s' \mid s,a)V(s') \right]
$$

with:

$$
R(s,a)=\mathbb E[\text{state score} \mid s,a] - \lambda C(a)
$$

Parameters:

| Parameter | Value |
|---|---:|
| Discount factor $\gamma$ | 0.85 |
| Cost weight $\lambda$ | 0.25 |
| Minimum observations per state-action | 3 |

The project solves this MDP using both **value iteration** and **policy iteration**.

## Constrained MDP

The constrained MDP is solved as a discounted occupancy-measure linear program. Let $x(s,a)$ be the discounted occupancy measure for state-action pair $(s,a)$.

Objective:

$$
\max_x \quad \sum_{s,a} x(s,a) r(s,a)
$$

Flow balance constraints:

$$
\sum_a x(s,a) - \gamma \sum_{s',a} x(s',a)P(s \mid s',a) = \mu_0(s), \quad \forall s
$$

Resource and risk constraints:

$$
(1-\gamma)\sum_{s,a}x(s,a)C(a) \le B
$$

$$
(1-\gamma)\sum_{s,a}x(s,a)P(\text{Low} \mid s,a) \le \alpha
$$

$$
(1-\gamma)\sum_{s,a}x(s,a)P(\text{Hit or Viral} \mid s,a) \ge \tau
$$

Default constrained MDP parameters:

| Parameter | Value |
|---|---:|
| Budget $B$ | 1.5 |
| Low-risk limit $\alpha$ | 0.2 |
| High-performance minimum $\tau$ | 0.4 |
| Minimum observations | 2 |

## Main Results

### Standard MDP Policy

Value iteration and policy iteration converge to the same deterministic policy:

| State | Recommended Action | Value | P(next High) | P(next Low) |
|---|---|---:|---:|---:|
| Low | Travel/Experience / ≤30 days | 6.958 | 0.076 | 0.441 |
| Normal | Lifestyle/Other / ≤30 days | 7.919 | 0.202 | 0.181 |
| Hit | Travel/Experience / ≤30 days | 11.020 | 0.705 | 0.007 |
| Viral | Travel/Experience / ≤14 days | 13.941 | 1.000 | 0.000 |


### Constrained MDP Policy

The constrained MDP is solved using a linear program. The solution status is **optimal**.

| Metric | Value |
|---|---:|
| Discounted average score | 1.627 |
| Discounted average cost | 1.494 |
| Discounted Low risk | 0.146 |
| Discounted high-performance rate | 0.485 |

The CMDP may return a randomized policy because occupancy-measure LP solutions can mix actions to satisfy resource and risk constraints. See `results/cmdp_occupancy_lp_policy.csv`.

## Key Interpretation

This final version is not just a descriptive Markov chain. It includes:

- empirical transition estimation;
- action-conditioned transition modeling;
- a reward function;
- value iteration;
- policy iteration;
- a constrained MDP solved by linear programming.

This makes the project a small but complete stochastic decision model.

## Limitations

1. The dataset contains only five creators and 200 videos.
2. The model identifies association, not causality.
3. Strategy costs are proxy costs, not real production budgets.
4. Action-conditioned transitions are sparse, so empirical-Bayes smoothing is used.
5. YouTube algorithm dynamics and external events are not directly observed.

## How to Run

```bash
pip install -r requirements.txt
python src/youtube_mdp_model.py
```

## Suggested Resume Description

Built a data-driven Markov Decision Process model for post-viral YouTube content strategy using 200 videos from five creators. Estimated empirical state-action transition probabilities, solved the discounted MDP using value iteration and policy iteration, and formulated a constrained MDP as an occupancy-measure linear program with resource and risk constraints.
