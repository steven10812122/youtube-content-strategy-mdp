
"""YouTube Content Strategy MDP.

This script reproduces the empirical Markov chain, value iteration, policy iteration,
and discounted constrained MDP occupancy-measure LP used in the project.
Run from the repository root:

    python src/youtube_mdp_model.py
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

STATES = ['Low', 'Normal', 'Hit', 'Viral']
STATE_SCORE = {'Low': 0.0, 'Normal': 1.0, 'Hit': 2.0, 'Viral': 3.0}
HIGH_STATES = {'Hit', 'Viral'}
DEADLINES = [7, 14, 30]
DISCOUNT = 0.85
LAMBDA_COST = 0.25
SMOOTHING_STRENGTH = 4.0
MIN_OBS = 3
STRATEGY_COST = {'Lifestyle/Other': 0.8, 'Talk/Topic': 1.0, 'Review/Tech': 1.2, 'Travel/Experience': 1.4}
DEADLINE_COST = {7: 0.6, 14: 0.3, 30: 0.1}
CMDP_BUDGET = 1.50
CMDP_LOW_RISK_LIMIT = 0.20
CMDP_HIGH_RATE_MIN = 0.40
CMDP_N_MIN = 2


def action_cost(action):
    strategy, deadline = action
    return STRATEGY_COST[strategy] + DEADLINE_COST[deadline]


def action_label(action):
    return f"{action[0]} / ≤{action[1]} days"


def estimate_state_action_model(transitions):
    states = STATES
    actions = [(a, d) for a in sorted(transitions['next_action_group'].unique()) for d in DEADLINES]
    counts = pd.crosstab(transitions['current_state'], transitions['next_state']).reindex(index=states, columns=states, fill_value=0)
    base_p = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    for s in states:
        if base_p.loc[s].sum() == 0:
            base_p.loc[s] = 1 / len(states)
    P, N = {}, {}
    rows = []
    score_vec = np.array([STATE_SCORE[s] for s in states])
    for s in states:
        for action in actions:
            strategy, dmax = action
            sub = transitions[(transitions.current_state == s) & (transitions.next_action_group == strategy) & (transitions.release_gap_days <= dmax)]
            n = len(sub)
            cnt = sub.next_state.value_counts().reindex(states, fill_value=0).astype(float).values
            prior = base_p.loc[s, states].values.astype(float)
            p = (cnt + SMOOTHING_STRENGTH * prior) / (n + SMOOTHING_STRENGTH)
            P[(s, action)] = p
            N[(s, action)] = n
            rows.append({
                'state': s, 'action': action_label(action), 'strategy': strategy, 'deadline_dmax': dmax,
                'sample_n': n, 'p_low': p[0], 'p_normal': p[1], 'p_hit': p[2], 'p_viral': p[3],
                'p_next_high': p[2] + p[3], 'expected_state_score': float(np.dot(p, score_vec)),
                'resource_cost': action_cost(action),
            })
    return actions, P, N, pd.DataFrame(rows), counts, base_p


def value_iteration(actions, P, N, gamma=DISCOUNT, lambda_cost=LAMBDA_COST, min_obs=MIN_OBS, tol=1e-8, max_iter=1000):
    score_vec = np.array([STATE_SCORE[s] for s in STATES])
    feasible = {s: [a for a in actions if N[(s, a)] >= min_obs] for s in STATES}
    for s in STATES:
        if not feasible[s]:
            feasible[s] = actions[:]
    V = np.zeros(len(STATES))
    hist = []
    policy = {}
    for it in range(max_iter):
        V_new = np.zeros_like(V)
        for si, s in enumerate(STATES):
            vals = []
            for a in feasible[s]:
                r = float(np.dot(P[(s, a)], score_vec) - lambda_cost * action_cost(a))
                vals.append((r + gamma * np.dot(P[(s, a)], V), a, r))
            V_new[si], policy[s], _ = max(vals, key=lambda x: x[0])
        residual = float(np.max(np.abs(V_new - V)))
        hist.append({'iteration': it, 'bellman_residual': residual})
        V = V_new
        if residual < tol:
            break
    rows = []
    for s in STATES:
        a = policy[s]
        p = P[(s, a)]
        rows.append({'state': s, 'recommended_action': action_label(a), 'strategy': a[0], 'deadline_dmax': a[1],
                     'sample_n': N[(s, a)], 'value_function': V[STATES.index(s)],
                     'p_next_high': p[2] + p[3], 'p_next_low': p[0],
                     'expected_state_score': float(np.dot(p, score_vec)), 'resource_cost': action_cost(a)})
    return pd.DataFrame(rows), pd.DataFrame(hist)


def policy_iteration(actions, P, N, gamma=DISCOUNT, lambda_cost=LAMBDA_COST, min_obs=MIN_OBS):
    score_vec = np.array([STATE_SCORE[s] for s in STATES])
    feasible = {s: [a for a in actions if N[(s, a)] >= min_obs] for s in STATES}
    for s in STATES:
        if not feasible[s]:
            feasible[s] = actions[:]
    policy = {s: feasible[s][0] for s in STATES}
    hist = []
    for it in range(100):
        Ppi = np.zeros((len(STATES), len(STATES)))
        rpi = np.zeros(len(STATES))
        for si, s in enumerate(STATES):
            a = policy[s]
            Ppi[si] = P[(s, a)]
            rpi[si] = np.dot(P[(s, a)], score_vec) - lambda_cost * action_cost(a)
        V = np.linalg.solve(np.eye(len(STATES)) - gamma * Ppi, rpi)
        changes = 0
        for si, s in enumerate(STATES):
            vals = []
            for a in feasible[s]:
                r = np.dot(P[(s, a)], score_vec) - lambda_cost * action_cost(a)
                vals.append((r + gamma * np.dot(P[(s, a)], V), a))
            best = max(vals, key=lambda x: x[0])[1]
            if best != policy[s]:
                policy[s] = best
                changes += 1
        hist.append({'iteration': it, 'policy_changes': changes})
        if changes == 0:
            break
    rows = []
    for s in STATES:
        a = policy[s]
        p = P[(s, a)]
        rows.append({'state': s, 'recommended_action': action_label(a), 'strategy': a[0], 'deadline_dmax': a[1],
                     'sample_n': N[(s, a)], 'value_function': V[STATES.index(s)],
                     'p_next_high': p[2] + p[3], 'p_next_low': p[0],
                     'expected_state_score': float(np.dot(p, score_vec)), 'resource_cost': action_cost(a)})
    return pd.DataFrame(rows), pd.DataFrame(hist)


def solve_cmdp(actions, P, N, transitions, gamma=DISCOUNT):
    score_vec = np.array([STATE_SCORE[s] for s in STATES])
    valid_sa = [(s, a) for s in STATES for a in actions if N[(s, a)] >= CMDP_N_MIN]
    var_idx = {sa: k for k, sa in enumerate(valid_sa)}
    m = len(valid_sa)
    mu0 = transitions.current_state.value_counts(normalize=True).reindex(STATES, fill_value=0).values.astype(float)
    Aeq = np.zeros((len(STATES), m)); beq = mu0.copy()
    for si, s in enumerate(STATES):
        for (sp, a), k in var_idx.items():
            if sp == s:
                Aeq[si, k] += 1
            Aeq[si, k] -= gamma * P[(sp, a)][si]
    c = np.zeros(m); cost = np.zeros(m); low = np.zeros(m); high = np.zeros(m); rew = np.zeros(m)
    for (s, a), k in var_idx.items():
        rew[k] = np.dot(P[(s, a)], score_vec)
        c[k] = -rew[k]
        cost[k] = action_cost(a)
        low[k] = P[(s, a)][0]
        high[k] = P[(s, a)][2] + P[(s, a)][3]
    Aub = np.array([(1-gamma)*cost, (1-gamma)*low, -(1-gamma)*high])
    bub = np.array([CMDP_BUDGET, CMDP_LOW_RISK_LIMIT, -CMDP_HIGH_RATE_MIN])
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)]*m, method='highs')
    if not res.success:
        return pd.DataFrame(), pd.DataFrame([{'status': 'infeasible', 'message': res.message}])
    x = res.x
    summary = pd.DataFrame([{'status': 'optimal', 'message': res.message,
                             'discounted_average_score': (1-gamma)*np.dot(rew, x),
                             'discounted_average_cost': (1-gamma)*np.dot(cost, x),
                             'discounted_low_risk': (1-gamma)*np.dot(low, x),
                             'discounted_high_rate': (1-gamma)*np.dot(high, x),
                             'budget': CMDP_BUDGET, 'low_risk_limit': CMDP_LOW_RISK_LIMIT,
                             'high_rate_min': CMDP_HIGH_RATE_MIN}])
    rows = []
    for s in STATES:
        inds = [var_idx[(s, a)] for a in actions if (s, a) in var_idx]
        occ = sum(x[i] for i in inds)
        for a in actions:
            if (s, a) not in var_idx:
                continue
            k = var_idx[(s, a)]
            pi = x[k] / occ if occ > 1e-12 else 0
            if pi > 1e-8:
                p = P[(s, a)]
                rows.append({'state': s, 'policy_probability': pi, 'action': action_label(a), 'strategy': a[0],
                             'deadline_dmax': a[1], 'sample_n': N[(s, a)], 'p_next_high': p[2]+p[3],
                             'p_next_low': p[0], 'expected_state_score': np.dot(p, score_vec),
                             'resource_cost': action_cost(a)})
    return pd.DataFrame(rows), summary


def run_pipeline(base_dir='.'):
    base = Path(base_dir)
    transitions = pd.read_csv(base/'data'/'transitions.csv')
    actions, P, N, sa_table, counts, base_p = estimate_state_action_model(transitions)
    vi_policy, vi_hist = value_iteration(actions, P, N)
    pi_policy, pi_hist = policy_iteration(actions, P, N)
    cmdp_policy, cmdp_summary = solve_cmdp(actions, P, N, transitions)
    return {
        'state_action_table': sa_table,
        'transition_counts': counts,
        'transition_probabilities': base_p,
        'value_iteration_policy': vi_policy,
        'value_iteration_history': vi_hist,
        'policy_iteration_policy': pi_policy,
        'policy_iteration_history': pi_hist,
        'cmdp_policy': cmdp_policy,
        'cmdp_summary': cmdp_summary,
    }

if __name__ == '__main__':
    results = run_pipeline('.')
    out = Path('results')
    out.mkdir(exist_ok=True)
    for name, df in results.items():
        df.to_csv(out/f'{name}.csv', index=(name.startswith('transition_')))
    print('Pipeline completed. Results written to results/.')
