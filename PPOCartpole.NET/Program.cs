using GymSharp;
using PPO.NET;
using System;

namespace PPOCartpole.NET
{
    internal class Program
    {
        private static dynamic step_state;
        private static double reward;
        private static double kl;

        static void Main(string[] args)
        {
            TestPPOEnv();

            Console.ReadKey();
        }

        private static void TestPPOEnv()
        {
            /// <summary>
            /// Hyperparameters
            /// </summary>

            // Hyperparameters of the PPO algorithm
            int stepsPerEpoch = 4000;
            int epochs = 30;
            int trainPolicyIterations = 80;
            int trainValueIterations = 80;
            double targetKl = (double)0.01;
            int[] hiddenSizes = { 64, 64 };

            // True if you want to render the environment
            bool render = false;

            /// <summary>
            /// Initializations
            /// </summary>

            // Initialize the environment and get the dimensionality of the
            // observation space and the number of possible actions
            Env env = new Env("CartPole-v1");
            Console.Clear();
            int observationDimensions = env.ObservationDimensions;
            int numActions = env.NumActions;

            // Initialize PPO
            PPOBinding ppo = new PPOBinding(observationDimensions,
                                            numActions,
                                            stepsPerEpoch,
                                            policyLearningRate: (double)3e-4,
                                            valueFunctionLearningRate: (double)1e-3,
                                            clipRatio: (double)0.2,
                                            hiddenSizes: hiddenSizes,
                                            gamma: (double)0.99,
                                            lam: (double)0.95);

            /// <summary>
            /// Load the model
            /// </summary>
            //ppo.Load("my_model");

            // Initialize the observation, episode return and episode length
            double[] observation = env.Reset();
            double episodeReturn = 0;
            int episodeLength = 0;

            /// <summary>
            /// Train
            /// </summary>
            // Iterate over the number of epochs
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Initialize the sum of the returns, lengths and number of episodes for each epoch
                double sumReturn = 0;
                double sumLength = 0;
                int numEpisodes = 0;

                // Iterate over the steps of each epoch
                for (int t = 0; t < stepsPerEpoch; t++)
                {
                    if (render)
                    {
                        env.Render();
                    }

                    // Get the logits, action, and take one step in the environment
                    //observation = observation.reshape(1, -1);

                    var sampledAction = ppo.SampleAction(observation);
                    double[][] logits = sampledAction.Item1;
                    int action = sampledAction.Item2;
                    (double[] observationNew, double reward, bool done) = env.Step(action);
                    episodeReturn += reward;
                    episodeLength += 1;

                    // Get the value and log-probability of the action
                    double valueT = ppo.Critic(observation);
                    double logProbabilityT = ppo.LogProbabilities(logits, action);

                    // Store obs, act, rew, v_t, logp_pi_t
                    ppo.buffer.Store(observation, action, reward, valueT, logProbabilityT);

                    // Update the observation
                    observation = observationNew;

                    // Finish trajectory if reached to a terminal state
                    bool terminal = done;
                    if (terminal || (t == stepsPerEpoch - 1))
                    {
                        double lastValue = done ? 0 : ppo.Critic(observation);
                        ppo.buffer.FinishTrajectory(lastValue);
                        sumReturn += episodeReturn;
                        sumLength += episodeLength;
                        numEpisodes += 1;
                        observation = env.Reset();
                        episodeReturn = 0;
                        episodeLength = 0;
                    }
                }

                // Get values from the buffer
                var (observationBuffer,
                    actionBuffer,
                    advantageBuffer,
                    returnBuffer,
                    logProbabilityBuffer) = ppo.buffer.Get();

                // Update the policy and implement early stopping using KL divergence
                double kl = 0;
                for (int i = 0; i < trainPolicyIterations; i++)
                {
                    kl = ppo.TrainPolicy(observationBuffer, actionBuffer, logProbabilityBuffer, advantageBuffer);
                    if (kl > 1.5 * targetKl)
                    {
                        Console.WriteLine($"Early stopping at iteration {i} due to reaching max kl: {kl:F2}");
                        break;
                    }
                }

                // Update the value function
                for (int i = 0; i < trainValueIterations; i++)
                {
                    ppo.TrainValueFunction(observationBuffer, returnBuffer);
                }

                // Print mean return and length for each epoch
                Console.WriteLine($"Epoch: {epoch + 1}. " +
                                  $"Mean Return: {sumReturn / numEpisodes:F2}. " +
                                  $"Mean Length: {sumLength / numEpisodes:F2}. " +
                                  $"KL Divergence: {kl:F2}");
            }

            /// <summary>
            /// Save the model
            /// </summary>
            //ppo.Save("my_model");
        }
    }
}