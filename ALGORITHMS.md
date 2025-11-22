📖 Multi-Armed Bandits (MAB) Algorithms Guide

This document provides a detailed overview of the Multi-Armed Bandit (MAB) algorithms implemented in the Full Multi-Armed Bandits library. The primary goal of any MAB algorithm is to optimally solve the Exploration vs. Exploitation dilemma: deciding whether to try less-known options (Exploration) or stick with the option that has historically proven to be the most rewarding (Exploitation).

## 1. Classical and Fundamental Algorithms

These algorithms form the bedrock of MAB theory and are most effective in stationary environments, where the true probability of reward for each arm remains constant over time.

### 1.1. Epsilon-Greedy (\epsilon-Greedy)

Mechanism (What it is):

The simplest and most intuitive MAB strategy. At each time step (t), the algorithm makes a probabilistic choice:
 * Exploitation (\mathbf{1 - \epsilon}): With a high probability, it selects the arm with the highest observed historical average reward (the "greedy" action).
 * Exploration (\mathbf{\epsilon}): With a small fixed probability (\epsilon, typically set between 0.01 and 0.1), it selects an arm uniformly at random.

Typical Uses:
 * Simple A/B testing frameworks where implementation complexity must be minimal.
 * Serving as a baseline performance metric for comparison against advanced algorithms.

Core Gain:
 * Simplicity and Guaranteed Exploration: It is trivial to implement and, since \epsilon is fixed, it ensures all arms are continuously sampled, preventing the algorithm from getting stuck in a local optimum forever.

### 1.2. UCB1 (Upper Confidence Bound 1)

Mechanism (What it is):

UCB1 is a deterministic algorithm based on the principle of optimism in the face of uncertainty. Instead of random exploration, it calculates an Upper Confidence Bound (UCB) index for each arm:
 * \bar{X}_i (Average Value): The Exploitation component (the average reward observed so far for arm i).
 * \sqrt{\frac{2 \ln(t)}{n_i}} (Confidence Term): The Exploration component. This term grows larger if the arm has been pulled few times (n_i is small) or if the total number of pulls (t) is high, thus forcing exploration on arms with high uncertainty.

Typical Uses:
 * Content personalization and dynamic pricing where regret minimization is critical.
 * Environments demanding highly efficient, log-time (logarithmic) exploration.

Core Gain:
 * Optimized Exploration: UCB only explores arms that have the potential to be truly optimal (those with high UCB indices), leading to better theoretical performance and lower cumulative regret compared to \epsilon-Greedy.

### 1.3. Thompson Sampling (TS)

Mechanism (What it is):

Thompson Sampling is a Bayesian approach. Instead of tracking a single point estimate (the average), it models the entire probability distribution of the true reward for each arm. For Bernoulli (0/1) rewards, it maintains a Beta distribution, defined by two parameters: \alpha (number of successes) and \beta (number of failures).
At each time step:
 * It samples one random probability value (a hypothesis of the true reward rate) from the Beta distribution of each arm.
 * It selects the arm whose sampled value is the highest.

Core Gain:
 * Natural Trade-Off: The width of the Beta distribution naturally represents uncertainty. Arms with few pulls have wide distributions, making them more likely to produce a high sample (Exploration). As an arm is pulled more, its distribution narrows around the true mean (Exploitation).
 * State-of-the-Art Performance: Often achieves the best empirical results in practice due to its probabilistic foundation.

## 2. Refined and Adaptive Algorithms

These algorithms enhance the classical strategies by dynamically adjusting the exploration rate or adapting to changes in the environment, moving beyond the simple stationary assumption.

### 2.1. Decaying Epsilon-Greedy

Mechanism (What it is):

This is an extension of \epsilon-Greedy where the exploration parameter \epsilon is not fixed, but decreases over time (t). Common decay schedules include exponential (\epsilon \propto \gamma^t) or inverse-proportional (\epsilon \propto 1/t). The goal is to maximize exploration early on when knowledge is low, and transition to pure exploitation later, once enough data has been collected to reliably estimate the true optimal arm.

Typical Uses:
 * A/B tests that run for a long, finite period where the true optimal choice is expected to remain constant after initial exploration.
 * Scenarios requiring faster convergence than fixed \epsilon-Greedy.

Core Gain:
 * Reduced Long-Term Regret: By minimizing the exploration rate later in the process, the algorithm drastically cuts down on cumulative regret, as less time is wasted pulling suboptimal arms unnecessarily.

### 2.2. Discounted UCB (\gamma-UCB)

Mechanism (What it is):

Designed specifically for Non-Stationary Bandits, where the true reward probability of an arm can change unpredictably. Discounted UCB modifies the standard UCB formula by applying an exponential decay factor (\gamma, where \gamma < 1) to the historical average reward estimate (\bar{X}_i). This mechanism forces the algorithm to gradually "forget" old rewards, giving recent rewards a higher influence on the current mean estimate. The confidence term remains similar, but now relies on the discounted effective number of pulls.

Typical Uses:
 * Financial modeling where market conditions change rapidly.
 * CTR optimization and news feed ranking where user preferences evolve quickly.
 * Any dynamic environment where stationarity cannot be assumed.

Core Gain:
 * Adaptability to Drift: Allows the algorithm to quickly detect and adapt to a shift in the optimal arm, minimizing the regret incurred after a change in the reward distribution.

### 2.3. UCB2


Mechanism (What it is):

UCB2 is a theoretical refinement of UCB1 that aims to be more efficient in how it allocates exploration time. It operates in blocks of pulls (l_i). An arm i is pulled l_i times consecutively, then the estimates are re-evaluated, and a new block size l'_i is calculated. This results in the algorithm pulling confident arms for longer periods before re-checking the confidence bound, which leads to tighter theoretical regret bounds.

Typical Uses:
 * Advanced research and implementation where low theoretical regret bounds are paramount.
 * Scenarios requiring high statistical efficiency in exploration.

Core Gain:
 * Improved Statistical Bounds: Offers theoretically better guarantees on cumulative regret than UCB1, especially when the reward gap between the optimal arm and suboptimal arms is very small.

## 3. Advanced Algorithms (Contextual and Adversarial)

These models extend the basic MAB framework to incorporate external information or handle hostile environments.

### 3.1. LinUCB (Linear Upper Confidence Bound)

Mechanism (What it is):

LinUCB addresses the Contextual Bandit problem. At each time step, the algorithm receives a context vector (\mathbf{x}_t, representing features like user demographics or time of day) before selecting an arm.
It assumes that the expected reward for selecting arm i is a linear function of the context: \mathbb{E}[\text{reward}_i] = \mathbf{x}_t \cdot \boldsymbol{\theta}_i.
The algorithm uses Ridge Regression to continually estimate the parameter vector (\boldsymbol{\theta}_i) for each arm. The selection is then based on the UCB principle, but applied to the predicted linear reward plus a confidence interval calculated in the feature space.

Typical Uses:
 * Personalized Recommendations: Choosing the right product (arm) for a specific user (context).
 * Targeted Advertising: Optimizing ad delivery based on a user's profile and browsing history.

Core Gain:
 * Personalization and Generalization: It allows the algorithm to make the best choice given the current situation, not just the best choice on average. It can also generalize knowledge from one context to another.

### 3.2. Exp3 (Exponential-Weight Algorithm for Exploration and Exploitation)

Mechanism (What it is):

Exp3 is designed for Adversarial Bandits, where the environment is assumed to be controlled by an "adversary" attempting to maximize the algorithm's cumulative regret. This is the worst-case scenario.
It maintains exponential weights over all arms. The arm selection is probabilistic based on these weights, coupled with a deliberate random exploration probability (\gamma) to protect against the adversary. The weight update uses a concept called the "estimated reward" or "virtual receipt" to normalize the impact of the observed reward.

Typical Uses:
 * Modeling competitive scenarios and complex, unknown systems (e.g., algorithmic trading).
 * Any situation where the reward generation mechanism is highly volatile or unpredictable.

Core Gain:
 * Maximum Robustness: Provides strong theoretical guarantees of low regret even under the most unpredictable (adversarial) reward sequences.
 * Guaranteed Performance: It offers a minimum guaranteed performance level that other greedy or optimistic methods may fail to achieve in adversarial settings.
