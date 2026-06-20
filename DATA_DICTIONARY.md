# Data Dictionary

## data/raw_videos.csv

| Column | Meaning |
|---|---|
| creator | YouTube channel name |
| video_order | Video order within creator sequence, from oldest to newest |
| publish_date | Publication date |
| title | Video title |
| content_type | Manually coded content type from the source dataset |
| views | View count observed at analysis date |

## data/processed_videos.csv

| Column | Meaning |
|---|---|
| days_since_upload | Days from publication date to analysis date |
| views_per_day | views divided by days_since_upload |
| creator_median_vpd | Creator-specific median views per day |
| ratio | views_per_day divided by creator_median_vpd |
| state | Low, Normal, Hit, or Viral |
| action_group | Simplified content strategy category |
| next_state | State of the next video by the same creator |

## data/transitions.csv

Each row is one within-creator transition from video t to video t+1.

| Column | Meaning |
|---|---|
| current_state | State of current video |
| next_state | State of next video |
| release_gap_days | Days between current and next video |
| next_action_group | Strategy category of the next video |
| next_is_high | 1 if next_state is Hit or Viral |
| next_is_low | 1 if next_state is Low |

## results/state_action_transition_model.csv

Estimated action-conditioned transition model.

| Column | Meaning |
|---|---|
| state | Current state |
| strategy | Next-video strategy |
| deadline_dmax | Maximum release deadline in days |
| sample_n | Number of matching observations |
| p_low, p_normal, p_hit, p_viral | Smoothed transition probabilities |
| p_next_high | p_hit + p_viral |
| expected_state_score | Expected next-state score |
| resource_cost | Proxy production/timing cost |
| immediate_reward | Expected score minus cost penalty |

## results/mdp_value_iteration_policy.csv

Optimal deterministic policy from value iteration.

## results/mdp_policy_iteration_policy.csv

Optimal deterministic policy from policy iteration.

## results/cmdp_occupancy_lp_policy.csv

Policy from the constrained MDP solved as a discounted occupancy-measure LP. The policy may be randomized.
