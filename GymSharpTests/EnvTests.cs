using Microsoft.VisualStudio.TestTools.UnitTesting;
using GymSharp;
using System;

namespace GymSharp.Tests
{
    [TestClass()]
    public class EnvTests
    {
        private Env env;

        [TestInitialize]
        public void Setup()
        {
            env = new Env("CartPole-v1");
        }

        [TestCleanup]
        public void Cleanup()
        {
            env.Dispose();
        }

        [TestMethod()]
        public void EnvTest()
        {
            Assert.IsNotNull(env);
        }

        [TestMethod()]
        public void ResetTest()
        {
            // Act
            var result = env.Reset();

            // Assert
            Assert.IsNotNull(result);
            Assert.IsInstanceOfType(result, typeof(double[]));
        }

        [TestMethod()]
        public void StepTest()
        {
            // Arrange
            int action = 0;

            env.Reset();

            // Act
            var (observation, reward, terminated) = env.Step(action);

            // Assert
            Assert.IsNotNull(observation);
            Assert.IsInstanceOfType(observation, typeof(double[]));
            Assert.IsNotNull(reward);
            Assert.IsInstanceOfType(reward, typeof(double));
            Assert.IsNotNull(terminated);
            Assert.IsInstanceOfType(terminated, typeof(bool));
        }

        [TestMethod()]
        public void RenderTest()
        {
            env.Reset();

            // Act
            env.Render();

            // Assert
            // No assertion needed for void methods
        }

        [TestMethod()]
        public void CloseTest()
        {
            // Act
            env.Close();

            // Assert
            // No assertion needed for void methods
        }

        [TestMethod()]
        public void DisposeTest()
        {
            // Act
            env.Dispose();

            // Assert
            // No assertion needed for void methods
        }
    }
}