using Microsoft.VisualStudio.TestTools.UnitTesting;
using PPO.NET;
using System;

namespace PPO.NET.Tests
{
    [TestClass()]
    public class CustomBufferTests
    {
        [TestMethod()]
        public void CustomBufferTest()
        {
            // Arrange
            CustomBuffer buffer = new CustomBuffer(observationDimensions: 2, size: 10);

            // Act

            // Assert
            Assert.IsNotNull(buffer);

            // Cleanup
            buffer.Dispose();
        }

        [TestMethod()]
        public void StoreTest()
        {
            // Arrange
            CustomBuffer buffer = new CustomBuffer(observationDimensions: 2, size: 10);
            double[] observation = new double[] { 1.0f, 2.0f };
            int action = 0;
            double reward = 0.0f;
            double value = 0.0f;
            double logProbability = 0.0f;

            // Act
            buffer.Store(observation, action, reward, value, logProbability);
            var (observationBuffer,
                    actionBuffer,
                    advantageBuffer,
                    returnBuffer,
                    logProbabilityBuffer) = buffer.Get();

            // Assert
            Assert.AreEqual(observation, observationBuffer[0]);

            // Cleanup
            buffer.Dispose();
        }

        [TestMethod()]
        public void FinishTrajectoryTest()
        {
            // Arrange
            CustomBuffer buffer = new CustomBuffer(observationDimensions: 2, size: 10);
            double lastValue = 1.0f;

            // Act
            buffer.FinishTrajectory(lastValue);
            var (observationBuffer,
                    actionBuffer,
                    advantageBuffer,
                    returnBuffer,
                    logProbabilityBuffer) = buffer.Get();

            // Assert
            Assert.AreEqual(lastValue, returnBuffer[0]);

            // Cleanup
            buffer.Dispose();
        }

        [TestMethod()]
        public void GetTest()
        {
            // Arrange
            CustomBuffer buffer = new CustomBuffer(observationDimensions: 2, size: 10);

            // Act
            var (observationBuffer,
                    actionBuffer,
                    advantageBuffer,
                    returnBuffer,
                    logProbabilityBuffer) = buffer.Get();

            // Assert
            Assert.IsNotNull(buffer);

            // Cleanup
            buffer.Dispose();
        }

        [TestMethod()]
        public void DisposeTest()
        {
            // Arrange
            CustomBuffer buffer = new CustomBuffer(observationDimensions: 2, size: 10);

            // Act
            buffer.Dispose();

            // Assert
            Assert.IsTrue(true);
        }
    }
}