using Python.Runtime;
using System;
using System.Data.Common;
using System.Linq;

namespace PPO.NET
{
    /// <summary>
    /// Class for the Proximal Policy Optimization (PPO) algorithm.
    /// </summary>
    public class PPOBinding
    {
        private PyObject pyPPO;
        private dynamic ppo;
        public CustomBuffer buffer;
        private dynamic np;
        private int trainPolicyIterations;
        private int trainValueIterations;
        private double targetKl;

        public PPOBinding(int observationDimensions,
                          int numActions,
                          int stepsPerEpoch,
                          int[] hiddenSizes = null,
                          double clipRatio = 0.2,
                          double policyLearningRate = 3e-4,
                          double valueFunctionLearningRate = 1e-3,
                          int trainPolicyIterations = 80,
                          int trainValueIterations = 80,
                          double targetKL = 0.01,
                          double gamma = 0.99,
                          double lam = 0.95)
        {
            if (hiddenSizes == null)
                hiddenSizes = new int[] { 64, 64 };

            this.trainPolicyIterations = trainPolicyIterations;
            this.trainValueIterations = trainValueIterations;
            this.targetKl = targetKL;

            buffer = new CustomBuffer(observationDimensions, stepsPerEpoch, gamma, lam);

            using (Py.GIL())
            {
                // Import the necessary Python modules
                dynamic site = Py.Import("site");
                dynamic sys = Py.Import("sys");
                np = Py.Import("numpy");
                site.addsitedir(@"C:\Users\Samuel\source\repos\PPOCartpole.NET\PPO.NET");

                // Create an instance of the PPO class
                dynamic ppoModule = Py.Import("ppo");
                pyPPO = ppoModule.PPO(new PyInt(observationDimensions),
                                      new PyInt(numActions),
                                      hiddenSizes,
                                      new PyFloat(clipRatio),
                                      new PyFloat(policyLearningRate),
                                      new PyFloat(valueFunctionLearningRate),
                                      new PyInt(trainPolicyIterations),
                                      new PyInt(trainValueIterations),
                                      new PyFloat(targetKL));
                this.ppo = pyPPO.AsManagedObject(typeof(object));
            }
        }

        /// <summary>
        /// Saves the given PPO object to a file specified by a path.
        /// </summary>
        /// <param name="path">The path of the file where the PPO object is to be saved.</param>
        public void Save(string path)
        {
            using (Py.GIL())
            {
                this.ppo.save(new PyString(path));
            }
        }

        /// <summary>
        /// Loads the PPO object from a file specified by a path.
        /// </summary>
        /// <param name="path">The path of the file where the PPO object is to be loaded from.</param>
        public void Load(string path)
        {
            using (Py.GIL())
            {
                this.ppo.load(new PyString(path));
            }
        }

        /// <summary>
        /// Samples an action from the policy given an observation.
        /// </summary>
        /// <param name="observation">The input observation vector to sample an action from.</param>
        /// <returns>A tuple containing the logits as a jagged double array.</returns>
        /// <returns>A double array containing the log probabilities.</returns>
        public (double[][], int) SampleAction(double[] observation)
        {
            using (Py.GIL())
            {
                // Convert the observation from a C# double array to a numpy array
                dynamic observationNp = np.array(observation.ToList<double>(), dtype: np.float32).reshape(1, -1);
                // Call the sample_action method and get the logits and action

                dynamic result = this.ppo.sample_action(observationNp);

                // Convert the logits from a numpy array to a C# double array
                double[][] logits = result[0].AsManagedObject(typeof(double[][]));

                // Convert the action from a numpy int64 to a C# int
                int action = result[1].AsManagedObject(typeof(int));

                return (logits, action);
            }
        }

        /// <summary>
        /// Get the action to take based on the given observation.
        /// </summary>
        /// <param name="observation">The observation to base the action on.</param>
        /// <returns>A tuple containing the action, its value and its log probability.</returns>
        public (int, double, double) GetAction(double[] observation)
        {
            using (Py.GIL())
            {
                // Convert the observation from a C# double array to a numpy array
                dynamic observationNp = np.array(observation.ToList<double>(), dtype: np.float32).reshape(1, -1);

                // Call the sample_action method and get the logits and action
                dynamic result = this.ppo.get_action(observationNp);

                // Convert the action from a numpy int64 to a C# int
                int action = result[0].AsManagedObject(typeof(int));

                // Convert the value from a numpy float64 to a C# double
                double value_t = result[1].AsManagedObject(typeof(double));

                // Convert the log-probabilities from a numpy array to a C# double
                double logprobability_t = result[2].AsManagedObject(typeof(double));

                return (action, value_t, logprobability_t);
            }
        }

        /// <summary>
        /// Computes the log probabilities of each action in the given logits and the action provided.
        /// </summary>
        /// <param name="logits">The logits from the model.</param>
        /// <param name="action">The action to compute log probability for in the given logits.</param>
        /// <returns>The log probability of the given action.</returns>
        public double LogProbabilities(double[][] logits, int actions)
        {
            using (Py.GIL())
            {
                // Convert the logits from a C# double array to a numpy array
                dynamic logitsNp = np.array(logits.Select(row => row.ToList()).ToList(), dtype: np.float32);

                // Convert the actions from a C# int array to a numpy array
                dynamic actionsNp = np.array(actions);

                // Call the logprobabilities method and get the log-probabilities
                dynamic logProbabilities = ppo.logprobabilities(logitsNp, actionsNp);

                // Convert the log-probabilities from a numpy array to a C# double
                double logProbabilitiesDouble = logProbabilities.AsManagedObject(typeof(double));

                return logProbabilitiesDouble;
            }
        }

        /// <summary>
        /// Trains the PPO policy on the given buffer.
        /// </summary>
        /// <param name="observationBuffer">The buffer containing the observations.</param>
        /// <param name="actionBuffer">The buffer containing the actions taken.</param>
        /// <param name="logprobabilityBuffer">The buffer containing the log probabilities of the actions.</param>
        /// <param name="advantageBuffer">The buffer containing the calculated advantage values.</param>
        /// <returns>The average loss for the policy network.</returns>
        public double TrainPolicy(double[][] observationBuffer, int[] actionBuffer, double[] logprobabilityBuffer, double[] advantageBuffer)
        {
            using (Py.GIL())
            {
                // Convert the observation buffer from a C# double array to a numpy array
                dynamic observationBufferNp = np.array(observationBuffer.Select(row => row.ToList()).ToList(), dtype: np.float32);

                // Convert the action buffer from a C# int array to a numpy array
                dynamic actionBufferNp = np.array(actionBuffer.ToList(), dtype: np.int64);

                // Convert the logprobability buffer from a C# double array to a numpy array
                dynamic logprobabilityBufferNp = np.array(logprobabilityBuffer.ToList(), dtype: np.float32);

                // Convert the advantage buffer from a C# double array to a numpy array
                dynamic advantageBufferNp = np.array(advantageBuffer.ToList(), dtype: np.float32);

                // Call the train_policy method and get the KL divergence
                dynamic kl = ppo.train_policy(observationBufferNp, actionBufferNp, logprobabilityBufferNp, advantageBufferNp);

                // Convert the KL divergence from a numpy float64 to a C# double
                double klDouble = kl.AsManagedObject(typeof(double));

                return klDouble;
            }
        }

        /// <summary>
        /// Trains the value function on the given buffer.
        /// </summary>
        /// <param name="observationBuffer">The buffer containing the observations.</param>
        /// <param name="returnBuffer">The buffer containing the calculated returns.</param>
        public void TrainValueFunction(double[][] observationBuffer, double[] returnBuffer)
        {
            using (Py.GIL())
            {
                // Convert the observation buffer from a C# double array to a numpy array
                dynamic observationBufferNp = np.array(observationBuffer.Select(row => row.ToList()).ToList(), dtype: np.float32);

                // Convert the return buffer from a C# double array to a numpy array
                dynamic returnBufferNp = np.array(returnBuffer.ToList(), dtype: np.float32);

                // Call the train_value_function method
                ppo.train_value_function(observationBufferNp, returnBufferNp);
            }
        }

        /// <summary>
        /// Calculates the value of the Critic network for a given observation.
        /// </summary>
        /// <param name="observation">The observation as a double array.</param>
        /// <returns>The Critic network value as a double.</returns>
        public double Critic(double[] observation)
        {
            using (Py.GIL())
            {
                // Convert the observation from a C# double array to a numpy array
                dynamic observationNp = np.array(observation.ToList(), dtype: np.float32).reshape(1, -1);

                // Call the Critic method and get the value
                dynamic value = ppo.critic(observationNp);

                // Convert the value from a numpy float64 to a C# double
                double valueDouble = value.AsManagedObject(typeof(double));

                return valueDouble;
            }
        }

        /// <summary>
        /// Trains the PPO model using the provided buffer of observations, actions, advantages, returns and log probabilities.
        /// </summary>
        /// <param name="observationBuffer">A 2D array containing the observations for each time step.</param>
        /// <param name="actionBuffer">An array containing the actions taken at each time step.</param>
        /// <param name="advantageBuffer">An array containing the advantages for each time step.</param>
        /// <param name="returnBuffer">An array containing the returns for each time step.</param>
        /// <param name="logProbabilityBuffer">An array containing the log probabilities for each time step.</param>
        public void Train(double[][] observationBuffer, int[] actionBuffer, double[] advantageBuffer, double[] returnBuffer, double[] logProbabilityBuffer)
        {
            double kl = 0;
            for (int i = 0; i < this.trainPolicyIterations; i++)
            {
                kl = TrainPolicy(observationBuffer, actionBuffer, logProbabilityBuffer, advantageBuffer);
                if (kl > 1.5 * this.targetKl)
                {
                    Console.WriteLine($"Early stopping at iteration {i} due to reaching max kl: {kl:F2}");
                    break;
                }
            }

            for (int i = 0; i < this.trainValueIterations; i++)
            {
                TrainValueFunction(observationBuffer, returnBuffer);
            }
        }
    }
}
