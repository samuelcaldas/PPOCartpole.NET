using System;
using System.Linq;

namespace PPO.NET
{
    /// <summary>
    /// Buffer for storing trajectories
    /// </summary>
    public class CustomBuffer : IDisposable
    {
        private readonly double gamma;
        private readonly double lam;
        private int pointer;
        private int trajectoryStartIndex;

        private readonly double[][] observationBuffer;
        private readonly int[] actionBuffer;
        private readonly double[] advantageBuffer;
        private readonly double[] rewardBuffer;
        private readonly double[] returnBuffer;
        private readonly double[] valueBuffer;
        private readonly double[] logProbabilityBuffer;

        private bool disposed = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="Buffer"/> class.
        /// </summary>
        /// <param name="observationDimensions">The number of observation dimensions.</param>
        /// <param name="size">The size of the buffer.</param>
        /// <param name="gamma">The <paramref name="gamma"/> parameter.</param>
        /// <param name="lam">The <paramref name="lam"/> parameter.</param>
        public CustomBuffer(int observationDimensions, int size, double gamma = 0.99f, double lam = 0.95f)
        {
            // Buffer initialization
            this.gamma = gamma;
            this.lam = lam;
            pointer = 0;
            trajectoryStartIndex = 0;

            observationBuffer = new double[size][];
            for (int i = 0; i < size; i++)
            {
                observationBuffer[i] = new double[observationDimensions];
            }

            actionBuffer = new int[size];
            advantageBuffer = new double[size];
            rewardBuffer = new double[size];
            returnBuffer = new double[size];
            valueBuffer = new double[size];
            logProbabilityBuffer = new double[size];
        }

        /// <summary>
        /// Appends one step of agent-environment interaction to the buffer.
        /// </summary>
        /// <param name="observation">The observation to store.</param>
        /// <param name="action">The action taken.</param>
        /// <param name="reward">The reward obtained.</param>
        /// <param name="value">The estimated value of the current state.</param>
        /// <param name="logProbability">The log probability of the action.</param>
        public void Store(double[] observation, int action, double reward, double value, double logProbability)
        {
            observation.CopyTo(observationBuffer[pointer], 0);
            actionBuffer[pointer] = action;
            rewardBuffer[pointer] = reward;
            valueBuffer[pointer] = value;
            logProbabilityBuffer[pointer] = logProbability;
            pointer++;
        }

        /// <summary>
        /// Completes the current trajectory by computing advantage estimates and rewards-to-go.
        /// </summary>
        /// <param name="lastValue">The last value of the trajectory.</param>
        public void FinishTrajectory(double lastValue = 0)
        {
            int pathLength = pointer - trajectoryStartIndex;
            double[] rewards = new double[pathLength + 1];
            double[] values = new double[pathLength + 1];

            Array.Copy(rewardBuffer, trajectoryStartIndex, rewards, 0, pathLength);
            rewards[pathLength] = lastValue;

            Array.Copy(valueBuffer, trajectoryStartIndex, values, 0, pathLength);
            values[pathLength] = lastValue;

            // Compute the GAE-Lambda advantage estimates
            double[] deltas = new double[pathLength];
            for (int i = 0; i < pathLength; i++)
            {
                deltas[i] = rewards[i] + gamma * values[i + 1] - values[i];
            }

            for (int i = trajectoryStartIndex; i < pointer; i++)
            {
                advantageBuffer[i] = DiscountedCumulativeSums(deltas, gamma * lam)[i - trajectoryStartIndex];
            }

            // Compute the rewards-to-go
            double[] returns = DiscountedCumulativeSums(rewards, gamma);
            for (int i = trajectoryStartIndex; i < pointer; i++)
            {
                returnBuffer[i] = returns[i - trajectoryStartIndex];
            }

            trajectoryStartIndex = pointer;
        }

        /// <summary>
        /// Gets all data of the buffer, normalizes the advantages, 
        /// and returns a tuple containing the observation buffer, 
        /// action buffer, normalized advantages, and log probability buffer.
        /// </summary>
        /// <returns>A tuple containing 4 arrays of doubles: observation buffer,
        /// action buffer, normalized advantages, and log probability buffer.</returns>
        public (double[][], int[], double[], double[], double[]) Get()
        {
            int bufferSize = pointer;
            pointer = 0;
            trajectoryStartIndex = 0;

            double advantageMean = advantageBuffer.Take(bufferSize).Average();
            double advantageStd = (double)Math.Sqrt(advantageBuffer.Take(bufferSize).Select(x => Math.Pow(x - advantageMean, 2)).Average());

            double[] normalizedAdvantages = new double[bufferSize];
            for (int i = 0; i < bufferSize; i++)
            {
                normalizedAdvantages[i] = (advantageBuffer[i] - advantageMean) / advantageStd;
            }

            return (observationBuffer,
                    actionBuffer,
                    normalizedAdvantages,
                    returnBuffer,
                    logProbabilityBuffer);
        }

        /// <summary>
        /// Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates.
        /// </summary>
        /// <param name="x">The vector to compute cumulative sums of.</param>
        /// <param name="discount">The discount rate.</param>
        /// <returns>The computed cumulative vectors.</returns>
        private static double[] DiscountedCumulativeSums(double[] x, double discount)
        {
            double[] y = new double[x.Length];
            double runningSum = 0;

            for (int i = x.Length - 1; i >= 0; i--)
            {
                runningSum = x[i] + discount * runningSum;
                y[i] = runningSum;
            }

            return y;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Dispose dos recursos gerenciados
                }

                // Dispose dos recursos não gerenciados
                for (int i = 0; i < observationBuffer.Length; i++)
                {
                    observationBuffer[i] = null;
                }
                // Removido as atribuições aos campos somente leitura
                // actionBuffer = null;
                // advantageBuffer = null;
                // rewardBuffer = null;
                // returnBuffer = null;
                // valueBuffer = null;
                // logProbabilityBuffer = null;

                disposed = true;
            }
        }

        ~CustomBuffer()
        {
            Dispose(false);
        }
    }
}