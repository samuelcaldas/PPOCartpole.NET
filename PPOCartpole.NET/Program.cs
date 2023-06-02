using GymSharp;
using PPO.NET;
using System;

namespace PPOCartpole.NET
{
    internal class Program
    {
        static void Main(string[] args)
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
            //Console.Clear();
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

            InteractionAgent agent = new InteractionAgent(env,
                                                          ppo,
                                                          stepsPerEpoch,
                                                          epochs,
                                                          trainPolicyIterations,
                                                          trainValueIterations,
                                                          targetKl,
                                                          hiddenSizes);
            agent.ExecuteTraining();

            Console.ReadKey();
        }
    }
}