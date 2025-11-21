/**
 * CORE MULTI-ARMED BANDITS LIBRARY (MAB)
 *
 * This file contains all standardized interfaces, linear algebra utilities,
 * and the implementation of various MAB algorithms.
 */

// --- 1. STANDARDIZED INTERFACES ---

type Vector = number[];
type Matrix = number[][];

/**
 * Interface for all Multi-Armed Bandit algorithms.
 * @template TContext The type of context (features) provided.
 */
export interface BanditAlgorithm<TContext = null> {
    /** The number of available arms (options). */
    readonly numArms: number;

    /**
     * Selects the next arm to pull based on the algorithm's strategy.
     * @param context The feature vector (context) for Contextual Bandits.
     * @returns The index (0 to numArms - 1) of the chosen arm.
     */
    selectArm(context?: TContext): number;

    /**
     * Updates the internal statistics of the pulled arm with the observed reward.
     * @param armIndex The index of the arm that was pulled.
     * @param reward The reward received (typically 0 or 1 for Bernoulli, but can be real numbers).
     * @param context The feature vector used in selection (required for LinUCB).
     */
    update(armIndex: number, reward: number, context?: TContext): void;

    /** Resets the algorithm's statistics. */
    reset(): void;

    /** Current estimates (for debugging/monitoring). */
    getEstimates?(): number[];
}

// --- 2. LINEAR ALGEBRA UTILITIES (For LinUCB) ---

const LinearAlgebra = {
    // Utility functions (zeros, identity, multiply, dot, inverse, etc.)
    zeros: (rows: number, cols: number): Matrix => Array(rows).fill(0).map(() => Array(cols).fill(0)),
    identity: (size: number): Matrix => Array(size).fill(0).map((_, i) => Array(size).fill(0).map((_, j) => (i === j ? 1 : 0))),
    addV: (V1: Vector, V2: Vector): Vector => V1.map((v, i) => v + V2[i]),
    dot: (V1: Vector, V2: Vector): number => V1.reduce((sum, v, i) => sum + v * V2[i], 0),

    multiply: (A: Matrix, B: Matrix): Matrix => {
        const rowsA = A.length;
        const colsA = A[0].length;
        const rowsB = B.length;

        if (colsA !== rowsB) throw new Error("Matrix A columns must equal Matrix B rows for multiplication.");

        const C = LinearAlgebra.zeros(rowsA, B[0].length);
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    },

    transposeV: (V: Vector): Matrix => V.map(v => [v]), // Vector to Column Matrix (d x 1)
    transposeVRow: (V: Vector): Matrix => [V], // Vector to Row Matrix (1 x d)

    /**
     * Basic Matrix Inverse implementation using Gauss-Jordan elimination.
     * NOTE: For production, a robust numerical library should be used.
     */
    inverse: (M: Matrix): Matrix => {
        if (M.length !== M[0].length) throw new Error("Matrix must be square for inversion.");
        const n = M.length;
        const identity = LinearAlgebra.identity(n);
        const augmented = M.map((row, i) => [...row, ...identity[i]]);

        for (let i = 0; i < n; i++) {
            // 1. Pivot selection (optional, but good practice)
            let pivot = i;
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(augmented[j][i]) > Math.abs(augmented[pivot][i])) pivot = j;
            }
            [augmented[i], augmented[pivot]] = [augmented[pivot], augmented[i]];

            // 2. Normalize pivot row
            const factor = augmented[i][i];
            if (Math.abs(factor) < 1e-10) throw new Error("Singular Matrix: Cannot be inverted.");
            for (let j = i; j < 2 * n; j++) {
                augmented[i][j] /= factor;
            }

            // 3. Eliminate other rows
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    const factor = augmented[j][i];
                    for (let k = i; k < 2 * n; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }

        // Return the right half of the augmented matrix
        return augmented.map(row => row.slice(n));
    }
};

// --- 3. BASE BANDIT CLASS ---

/**
 * Base class holding common state for MABs:
 * Counts of pulls and estimated average values.
 */
class BaseBandit {
    public counts: number[];
    public values: number[];

    constructor(public readonly numArms: number) {
        this.counts = new Array(numArms).fill(0);
        this.values = new Array(numArms).fill(0.0);
    }

    /**
     * Updates the estimated value using incremental average or exponential decay.
     */
    protected updateValue(armIndex: number, reward: number, decayFactor: number = 1.0): void {
        const n = this.counts[armIndex];
        const oldValue = this.values[armIndex];

        if (decayFactor < 1.0 && n > 0) {
            // Non-Stationary Update (Exponential Weighted Moving Average)
            this.values[armIndex] = oldValue * decayFactor + reward * (1 - decayFactor);
            this.counts[armIndex] = n + 1;
        } else {
            // Standard Stationary Update (Incremental Average)
            const newN = ++this.counts[armIndex];
            this.values[armIndex] = oldValue + (1 / newN) * (reward - oldValue);
        }
    }

    /** Finds the index of the arm with the highest estimate, tie-breaking randomly. */
    protected selectGreedy(estimates: number[]): number {
        let maxVal = -Infinity;
        let bestArms: number[] = [];

        for (let i = 0; i < this.numArms; i++) {
            const val = estimates[i];
            if (val > maxVal) {
                maxVal = val;
                bestArms = [i];
            } else if (val === maxVal) {
                bestArms.push(i);
            }
        }
        const randomIndex = Math.floor(Math.random() * bestArms.length);
        return bestArms[randomIndex];
    }

    public getEstimates(): number[] {
        return this.values;
    }

    public reset(): void {
        this.counts = new Array(this.numArms).fill(0);
        this.values = new Array(this.numArms).fill(0.0);
    }
}

// --- 4. ALGORITHM IMPLEMENTATIONS (CLASSICAL & REFINED) ---

/** Epsilon-Greedy: Explores with probability epsilon, exploits otherwise. */
export class EpsilonGreedy extends BaseBandit implements BanditAlgorithm<null> {
    private epsilon: number;
    constructor(numArms: number, epsilon: number = 0.1) {
        super(numArms);
        this.epsilon = epsilon;
    }
    selectArm(): number {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numArms); // Explore
        }
        return this.selectGreedy(this.values); // Exploit
    }
    update(armIndex: number, reward: number): void {
        this.updateValue(armIndex, reward);
    }
}

/** Decaying Epsilon-Greedy: Epsilon decreases exponentially over time to favor exploitation. */
export class DecayingEpsilonGreedy extends BaseBandit implements BanditAlgorithm<null> {
    private initialEpsilon: number;
    private decayRate: number; // e.g., 0.999

    constructor(numArms: number, initialEpsilon: number = 1.0, decayRate: number = 0.999) {
        super(numArms);
        this.initialEpsilon = initialEpsilon;
        this.decayRate = decayRate;
    }

    selectArm(): number {
        const totalTrials = this.counts.reduce((sum, count) => sum + count, 0);
        const currentEpsilon = this.initialEpsilon * Math.pow(this.decayRate, totalTrials);

        if (Math.random() < currentEpsilon) {
            return Math.floor(Math.random() * this.numArms);
        }
        return this.selectGreedy(this.values);
    }

    update(armIndex: number, reward: number): void {
        this.updateValue(armIndex, reward);
    }
}

/** UCB1: Upper Confidence Bound algorithm, optimizes exploration by penalizing less-pulled arms. */
export class UCB1 extends BaseBandit implements BanditAlgorithm<null> {
    private totalTrials: number;

    constructor(numArms: number) {
        super(numArms);
        this.totalTrials = 0;
    }

    selectArm(): number {
        this.totalTrials++;

        // Initialization phase
        for (let i = 0; i < this.numArms; i++) {
            if (this.counts[i] === 0) return i;
        }

        const ucbEstimates = this.values.map((v, i) => {
            const count = this.counts[i];
            // UCB Term: sqrt(2 * ln(t) / n_i)
            const confidenceTerm = Math.sqrt(2 * Math.log(this.totalTrials) / count);
            return v + confidenceTerm;
        });

        return this.selectGreedy(ucbEstimates);
    }

    update(armIndex: number, reward: number): void {
        this.updateValue(armIndex, reward);
    }

    reset(): void {
        super.reset();
        this.totalTrials = 0;
    }
}

/** Discounted UCB: Uses exponential decay (gamma) on rewards, suitable for non-stationary environments. */
export class DiscountedUCB extends UCB1 implements BanditAlgorithm<null> {
    private gamma: number; // Decay factor (e.g., 0.99)

    constructor(numArms: number, gamma: number = 0.99) {
        super(numArms);
        this.gamma = gamma;
    }

    update(armIndex: number, reward: number): void {
        // Use the discounted update logic
        this.updateValue(armIndex, reward, this.gamma);
    }

    // UCB1's selectArm is inherited and works by using the updated (discounted) 'values' and standard 'counts'.
}

/** Thompson Sampling: Bayesian approach using Beta distributions (for Bernoulli rewards) to model uncertainty. */
export class ThompsonSampling extends BaseBandit implements BanditAlgorithm<null> {
    private alpha: number[]; // Successes + 1
    private beta: number[];  // Failures + 1

    constructor(numArms: number) {
        super(numArms);
        this.alpha = new Array(numArms).fill(1); // Beta(1, 1) prior
        this.beta = new Array(numArms).fill(1);
    }

    /**
     * SIMULATED BETA SAMPLING.
     * Generates a sample from Beta(alpha, beta) distribution (simplified).
     */
    private sampleBeta(a: number, b: number): number {
        const mean = a / (a + b);
        // Add simulated Gaussian noise for demonstration purposes
        const noise = (Math.random() * 2 - 1) * 0.05;
        return Math.max(0, Math.min(1, mean + noise));
    }

    selectArm(): number {
        let maxSample = -Infinity;
        let bestArm = 0;

        for (let i = 0; i < this.numArms; i++) {
            const sample = this.sampleBeta(this.alpha[i], this.beta[i]);
            if (sample > maxSample) {
                maxSample = sample;
                bestArm = i;
            }
        }
        return bestArm;
    }

    update(armIndex: number, reward: number): void {
        // Thompson Sampling updates alpha (successes) and beta (failures)
        if (reward === 1) {
            this.alpha[armIndex]++;
        } else if (reward === 0) {
            this.beta[armIndex]++;
        }
    }

    public getEstimates(): number[] {
        // Return the mean of the Beta distribution for visualization
        return this.alpha.map((a, i) => a / (a + this.beta[i]));
    }

    reset(): void {
        this.alpha = new Array(this.numArms).fill(1);
        this.beta = new Array(this.numArms).fill(1);
    }
}

/** UCB2: Refined UCB, exploring in blocks for more focused information gathering. */
export class UCB2 extends BaseBandit implements BanditAlgorithm<null> {
    private totalTrials: number;
    private r: number[]; // Block count for arm 'i'

    constructor(numArms: number) {
        super(numArms);
        this.totalTrials = 0;
        this.r = new Array(numArms).fill(0);
    }

    selectArm(): number {
        this.totalTrials++;

        // Initialization phase
        for (let i = 0; i < this.numArms; i++) {
            if (this.counts[i] === 0) return i;
        }

        // UCB2 is more complex. For simplification, we use a block-based confidence interval.
        const ucb2Estimates = this.values.map((v, i) => {
            const Ti = this.counts[i];
            const ri = this.r[i];
            const alpha = 1 / (ri + 1);
            // Confidence term based on UCB2 formula
            const confidenceTerm = Math.sqrt((1 + alpha) * Math.log(Math.E / alpha) / (2 * Ti));
            return v + confidenceTerm;
        });

        return this.selectGreedy(ucb2Estimates);
    }

    update(armIndex: number, reward: number): void {
        // Standard update
        this.updateValue(armIndex, reward);

        // UCB2 block logic: Check if the arm count is a power of 2
        // A simple way to check if it's time to increment block (r_i)
        const Ti = this.counts[armIndex];
        const nextBlockSize = Math.ceil(Math.pow(2, this.r[armIndex] + 1));
        if (Ti >= nextBlockSize) {
            this.r[armIndex]++;
        }
    }

    reset(): void {
        super.reset();
        this.totalTrials = 0;
        this.r = new Array(this.numArms).fill(0);
    }
}

// --- 5. ADVANCED ALGORITHMS ---

/** LinUCB: Contextual Bandit using Ridge Regression and UCB confidence bounds. */
export class LinUCB implements BanditAlgorithm<Vector> {
    private readonly alpha: number;
    private readonly dimension: number;
    public readonly numArms: number;

    // A_i = D_i^T * D_i + I (Covariance matrix for Ridge Regression)
    private A: Matrix[];
    // b_i = D_i^T * c_i (Reward vector)
    private b: Vector[];

    constructor(numArms: number, dimension: number, alpha: number = 0.25) {
        this.numArms = numArms;
        this.dimension = dimension;
        this.alpha = alpha;
        this.A = [];
        this.b = [];
        this.reset();
    }

    selectArm(context: Vector): number {
        let maxUCB = -Infinity;
        let bestArm = 0;

        for (let i = 0; i < this.numArms; i++) {
            try {
                // 1. Calculate theta_i = A_i_inv * b_i (Ridge Regression coefficients)
                const A_i_inv = LinearAlgebra.inverse(this.A[i]);
                const theta_i = LinearAlgebra.multiply(A_i_inv, LinearAlgebra.transposeV(this.b[i])).map(r => r[0]);

                // 2. Estimated Reward: x_t . theta_i
                const estimatedReward = LinearAlgebra.dot(context, theta_i);

                // 3. Exploration Term: alpha * sqrt(x_t^T * A_i_inv * x_t)
                const x_t_T = LinearAlgebra.transposeVRow(context);
                const x_t = LinearAlgebra.transposeV(context);
                const P = LinearAlgebra.multiply(LinearAlgebra.multiply(x_t_T, A_i_inv), x_t)[0][0];
                const explorationTerm = this.alpha * Math.sqrt(P);

                // 4. UCB = Estimated Reward + Exploration Term
                const ucb = estimatedReward + explorationTerm;

                if (ucb > maxUCB) {
                    maxUCB = ucb;
                    bestArm = i;
                }
            } catch (error) {
                // In case of singular matrix (early steps), default to random exploration
                return Math.floor(Math.random() * this.numArms);
            }
        }
        return bestArm;
    }

    update(armIndex: number, reward: number, context: Vector): void {
        const x_t = LinearAlgebra.transposeV(context);
        const x_t_T = LinearAlgebra.transposeVRow(context);

        // 1. Update A_i: A_i += x_t * x_t^T
        const xx_T = LinearAlgebra.multiply(x_t, x_t_T);
        this.A[armIndex] = this.A[armIndex].map((row, r) => row.map((val, c) => val + xx_T[r][c]));

        // 2. Update b_i: b_i += reward * x_t
        const reward_x_t = context.map(x => x * reward);
        this.b[armIndex] = LinearAlgebra.addV(this.b[armIndex], reward_x_t);
    }

    reset(): void {
        this.A = [];
        this.b = [];
        for (let i = 0; i < this.numArms; i++) {
            // A_i initialized as Identity Matrix (lambda = 1)
            this.A.push(LinearAlgebra.identity(this.dimension));
            // b_i initialized as Zero Vector
            this.b.push(new Array(this.dimension).fill(0));
        }
    }
}

/** Exp3: Exponential-weight algorithm for Adversarial Bandits. */
export class Exp3 implements BanditAlgorithm<null> {
    public readonly numArms: number;
    private gamma: number; // Exploration factor
    private weights: number[];
    private probabilities: number[];
    private totalTrials: number;

    constructor(numArms: number, gamma: number = 0.1) {
        this.numArms = numArms;
        this.gamma = gamma;
        this.weights = new Array(numArms).fill(1.0);
        this.probabilities = new Array(numArms).fill(1 / numArms);
        this.totalTrials = 0;
    }

    selectArm(): number {
        this.totalTrials++;
        const totalWeight = this.weights.reduce((sum, w) => sum + w, 0);

        // 1. Calculate selection probabilities p_i
        for (let i = 0; i < this.numArms; i++) {
            // p_i = (1 - gamma) * (w_i / Sum(w)) + (gamma / K)
            this.probabilities[i] = (1 - this.gamma) * (this.weights[i] / totalWeight) + (this.gamma / this.numArms);
        }

        // 2. Sample from the distribution p
        let r = Math.random();
        let cumulativeProb = 0;
        for (let i = 0; i < this.numArms; i++) {
            cumulativeProb += this.probabilities[i];
            if (r < cumulativeProb) {
                return i;
            }
        }
        return this.numArms - 1; // Fallback
    }

    update(armIndex: number, reward: number): void {
        // 1. Calculate estimated reward (virtual receipt)
        // x_hat_i = reward / p_i
        const estimatedReward = reward / this.probabilities[armIndex];

        // 2. Update weight: w_i = w_i * exp( (gamma * x_hat_i) / K )
        const factor = Math.exp((this.gamma * estimatedReward) / this.numArms);
        this.weights[armIndex] *= factor;
    }

    public getEstimates(): number[] {
        return this.probabilities;
    }

    reset(): void {
        this.weights = new Array(this.numArms).fill(1.0);
        this.probabilities = new Array(this.numArms).fill(1 / this.numArms);
        this.totalTrials = 0;
    }
                 }
