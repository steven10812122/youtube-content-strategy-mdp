# Research Report

# YouTube Content Strategy under Uncertainty: A Markov Decision Process Approach

## 1. Motivation

Many online creators face a sequential decision problem after a video performs unusually well. A viral video may create short-term momentum, but the creator still needs to decide what kind of content to publish next and how quickly to publish it. This project treats that problem as a stochastic dynamic decision problem.

The central question is:

> Given the current video's performance state, which next-video strategy should be selected under uncertainty?

The project first estimates a Markov chain of video performance states and then extends the model into a Markov Decision Process by adding actions, rewards, and constraints.

## 2. Dataset

The dataset contains 200 long-form videos from five Taiwanese YouTube channels, with 40 videos per creator. Within each creator, adjacent videos define a transition. This gives 195 transitions.

## 3. State Space

Each video is classified into one of four states using creator-normalized views per day.

$$
r_i = \frac{\text{views per day of video } i}{\text{median views per day of the same creator}}
$$

| State | Definition | Score |
|---|---|---:|
| Low | $r_i < 0.7$ | 0 |
| Normal | $0.7 \le r_i < 2$ | 1 |
| Hit | $2 \le r_i < 5$ | 2 |
| Viral | $r_i \ge 5$ | 3 |

## 4. Baseline Markov Chain

Let $X_t$ be the state of the $t$-th video for a creator. The baseline Markov chain estimates:

$$
P(X_{t+1}=j \mid X_t=i)
$$

The estimated transition probabilities are stored in `results/markov_transition_probabilities.csv`.

## 5. MDP Formulation

The MDP is defined by:

- states $S=\{Low, Normal, Hit, Viral\}$;
- actions $A=\{(strategy, deadline)\}$;
- transition probabilities $P(s' \mid s,a)$;
- reward function $R(s,a)$;
- discount factor $\gamma=0.85$.

The reward is:

$$
R(s,a)=\mathbb E[\text{state score} \mid s,a] - \lambda C(a)
$$

with $\lambda=0.25$.

## 6. Action-Conditioned Transition Estimation

For each state-action pair, the model estimates:

$$
P(X_{t+1}=j \mid X_t=s,A_t=a)
$$

Because some state-action pairs have limited observations, the project uses empirical-Bayes smoothing:

$$
\hat P(j \mid s,a)=\frac{N_{saj}+\kappa \hat P(j\mid s)}{N_{sa}+\kappa}
$$

where $\kappa=4$.

## 7. Standard MDP Solution

The standard discounted MDP is solved by both value iteration and policy iteration.

Bellman optimality equation:

$$
V(s)=\max_a \left[ R(s,a)+\gamma \sum_{s'} P(s'\mid s,a)V(s') \right]
$$

### Value Iteration Policy

| State | Action | Value | P(next High) | P(next Low) |
|---|---|---:|---:|---:|
| Low | Travel/Experience / ≤30 days | 6.958 | 0.076 | 0.441 |
| Normal | Lifestyle/Other / ≤30 days | 7.919 | 0.202 | 0.181 |
| Hit | Travel/Experience / ≤30 days | 11.020 | 0.705 | 0.007 |
| Viral | Travel/Experience / ≤14 days | 13.941 | 1.000 | 0.000 |


Policy iteration returns the same deterministic policy, which verifies the dynamic programming solution.

## 8. Constrained MDP

The constrained MDP is solved using the discounted occupancy-measure linear programming formulation.

Objective:

$$
\max_x \sum_{s,a} x(s,a) r(s,a)
$$

Flow balance:

$$
\sum_a x(s,a)-\gamma \sum_{s',a} x(s',a)P(s\mid s',a)=\mu_0(s)
$$

Constraints:

$$
(1-\gamma)\sum_{s,a}x(s,a)C(a) \le B
$$

$$
(1-\gamma)\sum_{s,a}x(s,a)P(\text{Low}\mid s,a) \le \alpha
$$

$$
(1-\gamma)\sum_{s,a}x(s,a)P(\text{Hit or Viral}\mid s,a) \ge \tau
$$

The constrained MDP parameters are:

| Parameter | Value |
|---|---:|
| Budget $B$ | 1.5 |
| Low risk limit $\alpha$ | 0.2 |
| High-performance minimum $\tau$ | 0.4 |

The LP solution is **optimal** and achieves:

| Metric | Value |
|---|---:|
| Discounted average score | 1.627 |
| Discounted average cost | 1.494 |
| Discounted Low risk | 0.146 |
| Discounted high-performance rate | 0.485 |

## 9. Discussion

This project has three levels of modeling.

First, the empirical Markov chain describes how video performance states transition across a creator's video sequence.

Second, the standard MDP adds decisions and optimizes the expected discounted value of future video performance. This is solved using value iteration and policy iteration.

Third, the constrained MDP introduces resource and risk controls. Instead of always choosing the action with the highest expected score, the CMDP solves for a policy that respects a resource budget, limits Low-state risk, and maintains a minimum high-performance rate.

This is why the final project is no longer just a YouTube data analysis. It is a stochastic decision model with a state space, action space, transition model, reward function, constraints, and optimized policies.

## 10. Limitations

- The dataset is small and creator-specific.
- The model estimates association, not causality.
- Production cost is proxied by strategy and deadline, not directly observed.
- Some action-conditioned transitions are sparse.
- YouTube recommendation algorithms and external events are not observed.

## 11. Conclusion

The project demonstrates how a content-performance problem can be formulated as a Markov Decision Process. The final model includes empirical transition estimation, dynamic programming, and a constrained MDP solved by linear programming. The empirical results suggest post-viral persistence, while the MDP framework provides a structured way to choose next-video strategies under uncertainty.
