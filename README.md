![Full Multi-Armed Bandits (MAB)](https://i.imgur.com/Z0xQPXF.png) 

# Full Multi-Armed Bandits (MAB) Library

A comprehensive, zero-dependency TypeScript library providing standardized implementations of various Multi-Armed Bandit (MAB) algorithms, suitable for A/B testing, personalization, and reinforcement learning research.
This library includes classical, refined, non-stationary, and contextual bandit algorithms, all adhering to a single, simple BanditAlgorithm interface.

## 🚀 Features

 * Standardized Interface: Easily swap algorithms using the BanditAlgorithm<TContext> interface.
 * Zero Dependencies: Core logic and minimal linear algebra utilities are included for autonomy.
 * TypeScript Focused: Strong typing and clear structure for robust application development.
 * Full Coverage: Includes advanced algorithms like LinUCB (Contextual) and Exp3 (Adversarial).

## 📦 Installation

To set up the project locally:

### Clone the repository (or download the files)

```
git clone [https://github.com/YourOrganization/full-multi-armed-bandits.git](https://github.com/YourOrganization/full-multi-armed-bandits.git)
cd full-multi-armed-bandits
```

### Install dependencies (TypeScript and ts-node for running examples)

```
npm install
```

### Build the project (compiles src/index.ts to dist/index.js)

```
npm run build
```

## 🧠 Algorithms Included

| Category | Algorithm | Description | Usage Scenario |
|---|---|---|---|
| Classical | EpsilonGreedy | Simple, time-agnostic exploration with fixed probability (\epsilon). | Quick A/B testing, stable environments. |
| Classical | UCB1 | Upper Confidence Bound. Optimizes exploration by penalizing less-pulled arms. | Stable, low-regret requirements. |
| Bayesian | ThompsonSampling | Uses a Bayesian approach (Beta distribution) to model uncertainty and sample the best arm. | Excellent performance, highly probabilistic selection. |
| Refined | DecayingEpsilonGreedy | Reduces \epsilon over time, favoring exploitation as certainty increases. | Faster convergence in stable environments. |
| Refined | UCB2 | Explores in blocks, a refinement of UCB1 suitable for complex probability distributions. | Theoretical robustness, advanced analysis. |
| Non-Stationary | DiscountedUCB | Weights recent rewards more heavily (\gamma), allowing the algorithm to "forget" old data. | Click-Through Rate (CTR) analysis where user preferences change. |
| Adversarial | Exp3 | Exponential-weight algorithm designed to minimize regret against a malicious adversary. | Unknown environments, robust against deliberate reward manipulation. |
| Contextual | LinUCB | Linear Upper Confidence Bound. Uses context (features) to inform arm selection via Ridge Regression. | Personalized recommendations, targeted advertising. |

##💡 Usage Examples

The src/examples.ts file demonstrates the simulation of all algorithms across three distinct environments (Classical, Non-Stationary, and Contextual).
To run the examples and compare algorithm performance:

```
npm start
```

### Example Output Snippet (Performance Comparison)

Running the simulation for 5,000 trials on a Classical Stationary Bandit environment (Optimal Arm Probability: 0.70):

| ALGORITHM | Total Reward | Average Reward | Cumulative Regret |
|---|---|---|---|
| Epsilon-Greedy (ε=0.1) | 4801 | 0.9602 | 100.21 |
| UCB1 | 4910 | 0.9820 | 45.89 |
| Thompson Sampling | 4905 | 0.9810 | 50.11 |
| Decaying Epsilon-Greedy | 4880 | 0.9760 | 69.30 |

(Note: Results are stochastic and will vary slightly on each run.)

Code Example: Implementing a Simple Bandit

```
import { EpsilonGreedy, BanditAlgorithm } from './index';

// 1. Initialize the algorithm
const numArms = 3;
const epsilon = 0.1;
const myBandit: BanditAlgorithm<null> = new EpsilonGreedy(numArms, epsilon);

// 2. Define the true environment (hidden from the bandit)
const trueProbs = [0.2, 0.8, 0.5]; // Arm 1 is the best

for (let trial = 0; trial < 100; trial++) {
    // Select the arm
    const armIndex = myBandit.selectArm();

    // Pull the arm (get a reward)
    const reward = Math.random() < trueProbs[armIndex] ? 1 : 0;

    // Update the bandit's knowledge
    myBandit.update(armIndex, reward);
}

console.log("Final Estimates:", myBandit.getEstimates());
// Expected: Estimates should be close to [0.2, 0.8, 0.5]

```

## 🤝 Contributing

Contributions are highly welcome! Feel free to open issues or pull requests for:
 * Implementing more advanced MAB algorithms (e.g., LinTS, more robust Beta samplers).
 * Improving the efficiency of the Linear Algebra utilities.
 * Adding more comprehensive testing.



## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
