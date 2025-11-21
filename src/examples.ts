/**
 * MAB USAGE EXAMPLES
 * Demonstrates how to use all implemented MAB algorithms
 * in different simulation environments.
 */

import {
    BanditAlgorithm,
    EpsilonGreedy,
    DecayingEpsilonGreedy,
    UCB1,
    DiscountedUCB,
    UCB2,
    ThompsonSampling,
    LinUCB,
    Exp3
} from './index';

type Vector = number[];

// --- 1. CLASSICAL BANDIT ENVIRONMENT ---

/** Simulates a standard Bernoulli Multi-Armed Bandit environment. */
class BanditEnvironment {
    private probabilities: number[];
    public totalSteps: number;
    public totalReward: number;

    constructor(probabilities: number[]) {
        this.probabilities = probabilities;
        this.totalSteps = 0;
        this.totalReward = 0;
    }

    /** Pulls an arm and returns a Bernoulli reward (1 or 0). */
    pullArm(armIndex: number): number {
        const prob = this.probabilities[armIndex];
        const reward = Math.random() < prob ? 1 : 0;
        this.totalSteps++;
        this.totalReward += reward;
        return reward;
    }

    /** Runs the simulation for classical MABs. */
    runSimulation(algorithm: BanditAlgorithm<null>, numTrials: number): { totalReward: number, cumulativeRegret: number } {
        this.totalSteps = 0;
        this.totalReward = 0;
        algorithm.reset();

        const maxProb = Math.max(...this.probabilities);
        let cumulativeRegret = 0;

        for (let t = 1; t <= numTrials; t++) {
            const arm = algorithm.selectArm();
            const reward = this.pullArm(arm);
            algorithm.update(arm, reward);

            // Regret: (Expected reward of optimal arm) - (Expected reward of chosen arm)
            const regret = maxProb - this.probabilities[arm];
            cumulativeRegret += regret;
        }

        return { totalReward: this.totalReward, cumulativeRegret };
    }
}


// --- 2. CONTEXTUAL BANDIT ENVIRONMENT (For LinUCB) ---

/** Simulates a Contextual Bandit environment where reward depends on context (features). */
class ContextualBanditEnvironment {
    private trueTheta: Matrix; // True linear coefficients (dimension x numArms)
    private readonly dimension: number;
    public readonly numArms: number;

    constructor(numArms: number, dimension: number) {
        this.numArms = numArms;
        this.dimension = dimension;
        // Initialize true linear coefficients randomly
        this.trueTheta = Array(dimension).fill(0).map(() =>
            Array(numArms).fill(0).map(() => Math.random() * 2 - 1)
        );
    }

    /** Generates a random context vector. */
    generateContext(): Vector {
        return Array(this.dimension).fill(0).map(() => Math.random() * 2 - 1);
    }

    /** Pulls an arm in the given context (simplified linear model). */
    pullArm(armIndex: number, context: Vector): number {
        // Expected Reward = context . trueTheta_i
        const trueTheta_i = this.trueTheta.map(row => row[armIndex]);
        const expectedReward = context.reduce((sum, x, i) => sum + x * trueTheta_i[i], 0);

        // Add noise and clip for Bernoulli reward approximation
        const finalRewardProb = Math.min(1, Math.max(0, expectedReward * 0.5 + 0.5)); // Map to [0, 1]
        return Math.random() < finalRewardProb ? 1 : 0;
    }

    /** Runs the simulation for Contextual MABs. */
    runSimulation(algorithm: BanditAlgorithm<Vector>, numTrials: number): { totalReward: number, cumulativeRegret: number } {
        algorithm.reset();
        let totalReward = 0;
        let cumulativeRegret = 0;

        for (let t = 1; t <= numTrials; t++) {
            const context = this.generateContext();

            // Find optimal arm for regret calculation
            let maxExpectedReward = -Infinity;
            let optimalArmExpectedReward = 0;
            for (let i = 0; i < this.numArms; i++) {
                const trueTheta_i = this.trueTheta.map(row => row[i]);
                const expected = context.reduce((sum, x, j) => sum + x * trueTheta_i[j], 0);
                if (expected > maxExpectedReward) {
                    maxExpectedReward = expected;
                }
            }

            // Select arm, get reward, and update
            const selectedArm = algorithm.selectArm(context);
            const reward = this.pullArm(selectedArm, context);
            algorithm.update(selectedArm, reward, context);

            // Regret: (Optimal Expected Reward) - (Selected Expected Reward)
            const selectedArmTrueTheta = this.trueTheta.map(row => row[selectedArm]);
            const selectedArmExpectedReward = context.reduce((sum, x, i) => sum + x * selectedArmTrueTheta[i], 0);
            
            const regret = maxExpectedReward - selectedArmExpectedReward;
            cumulativeRegret += regret;
            totalReward += reward;
        }

        return { totalReward, cumulativeRegret };
    }
}


// --- 3. DEMONSTRATION LOGIC ---

// Helper to display results
function displayResults<TContext>(name: string, algorithm: BanditAlgorithm<TContext>, result: { totalReward: number, cumulativeRegret: number }, numTrials: number) {
    console.log(`\n============================================`);
    console.log(`ALGORITHM: ${name}`);
    console.log(`Total Trials: ${numTrials}`);
    console.log(`Total Reward: ${result.totalReward}`);
    console.log(`Average Reward: ${(result.totalReward / numTrials).toFixed(4)}`);
    console.log(`Cumulative Regret: ${result.cumulativeRegret.toFixed(4)}`);

    if (algorithm.getEstimates) {
        const estimates = algorithm.getEstimates();
        console.log(`Final Estimates (Avg/Prob): [${estimates.map(e => e.toFixed(4)).join(', ')}]`);
        if ('counts' in algorithm && Array.isArray((algorithm as any).counts)) {
            console.log(`Final Counts: [${(algorithm as any).counts.join(', ')}]`);
        }
    } else {
         console.log(`Final Estimates: (Contextual Algorithm - estimates depend on input context)`);
    }
    console.log(`============================================`);
}

// --- RUN SIMULATIONS ---
function runAllExamples() {
    console.log("--- FULL MULTI-ARMED BANDITS LIBRARY DEMO ---");
    const numArms = 5;
    const numTrials = 5000;
    const armProbabilities = [0.15, 0.25, 0.70, 0.50, 0.60]; // Optimal Arm is 2 (0.70)
    const environment = new BanditEnvironment(armProbabilities);

    console.log(`\n--- SCENARIO A: CLASSICAL STATIONARY BANDITS (K=${numArms}, T=${numTrials}) ---`);
    console.log(`True Probabilities: [${armProbabilities.join(', ')}]`);

    // 1. Classical Algorithms
    const eg = new EpsilonGreedy(numArms, 0.1);
    const resEg = environment.runSimulation(eg, numTrials);
    displayResults("1. Epsilon-Greedy (ε=0.1)", eg, resEg, numTrials);

    const deg = new DecayingEpsilonGreedy(numArms, 1.0, 0.999);
    const resDeg = environment.runSimulation(deg, numTrials);
    displayResults("2. Decaying Epsilon-Greedy (Decays to 0)", deg, resDeg, numTrials);

    const ucb1 = new UCB1(numArms);
    const resUcb1 = environment.runSimulation(ucb1, numTrials);
    displayResults("3. UCB1", ucb1, resUcb1, numTrials);

    const ts = new ThompsonSampling(numArms);
    const resTs = environment.runSimulation(ts, numTrials);
    displayResults("4. Thompson Sampling (Bayesian)", ts, resTs, numTrials);

    const ucb2 = new UCB2(numArms);
    const resUcb2 = environment.runSimulation(ucb2, numTrials);
    displayResults("5. UCB2 (Block-Based Exploration)", ucb2, resUcb2, numTrials);

    const exp3 = new Exp3(numArms, 0.1);
    const resExp3 = environment.runSimulation(exp3, numTrials);
    displayResults("6. Exp3 (Adversarial, γ=0.1)", exp3, resExp3, numTrials);

    // 2. Non-Stationary Scenario (Discounted UCB vs. UCB1)
    console.log(`\n--- SCENARIO B: NON-STATIONARY BANDITS (γ=0.99) ---`);
    console.log(`(Environment is set to have a probability shift after 50% of trials, favoring adaptive algorithms)`);

    const nonStationaryEnv = new BanditEnvironment(armProbabilities);
    // Overriding pullArm to simulate a shift (Arm 4 becomes optimal after 2500 steps)
    nonStationaryEnv.pullArm = (armIndex: number): number => {
        let currentProbabilities = armProbabilities;
        if (nonStationaryEnv.totalSteps > numTrials / 2) {
            currentProbabilities = [0.1, 0.2, 0.5, 0.8, 0.6]; // New optimal arm is 3 (0.8)
        }
        const prob = currentProbabilities[armIndex];
        const reward = Math.random() < prob ? 1 : 0;
        nonStationaryEnv.totalSteps++;
        nonStationaryEnv.totalReward += reward;
        return reward;
    };

    const ucb1_ns = new UCB1(numArms);
    const resUcb1Ns = nonStationaryEnv.runSimulation(ucb1_ns, numTrials);
    displayResults("7. UCB1 (Non-Adaptive, in Non-Stationary Env)", ucb1_ns, resUcb1Ns, numTrials);

    const discountedUcb = new DiscountedUCB(numArms, 0.99);
    const resDiscUcb = nonStationaryEnv.runSimulation(discountedUcb, numTrials);
    displayResults("8. Discounted UCB (Adaptive, γ=0.99)", discountedUcb, resDiscUcb, numTrials);


    // 3. Contextual Scenario (LinUCB)
    const dimension = 4; // Context vector dimension
    const numTrialsContextual = 5000;
    const contextualEnv = new ContextualBanditEnvironment(numArms, dimension);

    console.log(`\n--- SCENARIO C: CONTEXTUAL BANDITS (D=${dimension}, K=${numArms}, T=${numTrialsContextual}) ---`);

    const linUcb = new LinUCB(numArms, dimension, 0.5);
    const resLinUcb = contextualEnv.runSimulation(linUcb, numTrialsContextual);
    displayResults("9. LinUCB (Contextual Bandit, α=0.5)", linUcb, resLinUcb, numTrialsContextual);
}

runAllExamples(); 
