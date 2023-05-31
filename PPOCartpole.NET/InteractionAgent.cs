using GymSharp;
using PPO.NET;
using PPOCartpole.NET;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace PPOCartpole.NET
{
    public class InteractionAgent
    {
        private Env env;
        private PPOBinding ppo;
        private int stepsPerEpoch;
        private int epochs;
        private int trainPolicyIterations;
        private int trainValueIterations;
        private double targetKl;
        private int[] hiddenSizes;

        public InteractionAgent(Env env, PPOBinding ppo, int stepsPerEpoch, int epochs, int trainPolicyIterations, int trainValueIterations, double targetKl, int[] hiddenSizes)
        {
            this.env = env;
            this.ppo = ppo;
            this.stepsPerEpoch = stepsPerEpoch;
            this.epochs = epochs;
            this.trainPolicyIterations = trainPolicyIterations;
            this.trainValueIterations = trainValueIterations;
            this.targetKl = targetKl;
            this.hiddenSizes = hiddenSizes;
        }

        public void Train()
        {
            double[] observation = env.Reset();
            double episodeReturn = 0;
            int episodeLength = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double sumReturn = 0;
                double sumLength = 0;
                int numEpisodes = 0;

                for (int t = 0; t < stepsPerEpoch; t++)
                {
                    var sampledAction = ppo.SampleAction(observation);
                    double[][] logits = sampledAction.Item1;
                    int action = sampledAction.Item2;

                    (double[] observationNew, double reward, bool done) = env.Step(action);
                    episodeReturn += reward;
                    episodeLength += 1;

                    double valueT = ppo.Critic(observation);
                    double logProbabilityT = ppo.LogProbabilities(logits, action);

                    ppo.buffer.Store(observation, action, reward, valueT, logProbabilityT);

                    observation = observationNew;

                    bool terminal = done || (t == stepsPerEpoch - 1);
                    if (terminal)
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

                var (observationBuffer, actionBuffer, advantageBuffer, returnBuffer, logProbabilityBuffer) = ppo.buffer.Get();

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

                for (int i = 0; i < trainValueIterations; i++)
                {
                    ppo.TrainValueFunction(observationBuffer, returnBuffer);
                }
            }
        }
    }
}