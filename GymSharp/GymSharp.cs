using System;
using Python.Runtime;

namespace GymSharp
{
    public class Env : IDisposable
    {
        private dynamic env;
        private dynamic np;
        private PyObject pyEnv;
        public int ObservationDimensions
        {
            get
            {
                using (Py.GIL())
                {
                    return env.observation_space.shape[0];
                }
            }
        }

        public int NumActions
        {
            get
            {
                using (Py.GIL())
                {
                    return env.action_space.n;
                }
            }
        }

        public Env(string environment)
        {
            using (Py.GIL())
            {
                dynamic gym = Py.Import("gym");
                pyEnv = gym.make(environment);
                env = pyEnv.AsManagedObject(typeof(object));
                np = Py.Import("numpy");
            }
        }

        /// <summary>
        /// Performs one step of the environment's dynamics.
        /// </summary>
        /// <param name="action">An integer representing the action to take. 0 represents pushing the cart to the left and 1 represents pushing the cart to the right.</param>
        /// <returns>
        /// A tuple containing the next observation, reward, termination flag, and additional information.
        /// </returns>
        /// <remarks>
        /// <para>Observation: A double array with shape (4,) representing the cart position, cart velocity, pole angle, and pole angular velocity.</para>
        /// <para>Reward: A double representing the reward for taking the given action. A reward of +1 is given for every step taken, including the termination step.</para>
        /// <para>Termination flag: A boolean indicating whether or not the episode has ended. The episode ends if any of the following occurs:</para>
        /// <list type="number">
        /// <item>Termination: Pole Angle is greater than ±12°</item>
        /// <item>Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)</item>
        /// <item>Truncation: Episode length is greater than 500 (200 for v0)</item>
        /// </list>
        /// <para>Additional information: An object containing additional information about the environment's state.</para>
        /// </remarks>
        public (double[], double, bool) Step(int action)
        {
            using (Py.GIL())
            {
                var result = env.step(action);

                double[] observation = result[0].AsManagedObject(typeof(double[]));
                double reward = result[1].AsManagedObject(typeof(double));
                bool terminated = result[2].AsManagedObject(typeof(bool));

                return (observation, reward, terminated);
            }
        }

        public double[] Reset(int Seed=0)
        {
            using (Py.GIL())
            {
                var result = env.reset();

                double[] observation = result[0].AsManagedObject(typeof(double[]));

                return observation;
            }
        }

        public void Render()
        {
            using (Py.GIL())
            {
                env.render();
            }
        }

        public void Close()
        {
            using (Py.GIL())
            {
                env.close();
            }
        }

        public void Dispose()
        {
            pyEnv.Dispose();
        }
    }
}
