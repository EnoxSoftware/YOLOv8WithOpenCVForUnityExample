using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker
{
    public interface IProcessingWorker : IDisposable
    {
        /// <summary>
        /// Sets the input Mat at the specified index.
        /// </summary>
        /// <param name="input">The input Mat. Can be null to clear the input.</param>
        /// <param name="index">The index at which to set the input.</param>
        void SetInput(Mat input, int index = 0);

        /// <summary>
        /// Sets multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat. Can be null to clear the input.</param>
        /// <param name="input2">The second input Mat. Can be null to clear the input.</param>
        void SetInputs(Mat input1, Mat input2);

        /// <summary>
        /// Sets multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat. Can be null to clear the input.</param>
        /// <param name="input2">The second input Mat. Can be null to clear the input.</param>
        /// <param name="input3">The third input Mat. Can be null to clear the input.</param>
        void SetInputs(Mat input1, Mat input2, Mat input3);

        /// <summary>
        /// Sets multiple input Mats.
        /// </summary>
        /// <param name="inputs">The input Mats. Can be null to clear all inputs. Individual elements can also be null.</param>
        void SetInputs(params Mat[] inputs);

        /// <summary>
        /// Returns a reference (submat) to the output Mat at the specified index.
        /// </summary>
        /// <param name="index">The output index.</param>
        /// <returns>The output Mat.</returns>
        Mat PeekOutput(int index = 0);

        /// <summary>
        /// Returns a copy of the output Mat at the specified index.
        /// </summary>
        /// <param name="index">The output index.</param>
        /// <returns>A new Mat containing a copy of the output.</returns>
        Mat CopyOutput(int index = 0);

        /// <summary>
        /// Copies the output Mat at the specified index to a destination Mat.
        /// </summary>
        /// <param name="dst">The destination Mat.</param>
        /// <param name="index">The output index.</param>
        void CopyOutputTo(Mat dst, int index = 0);

        /// <summary>
        /// Visualizes the processing result on the given image.
        /// Should be overridden in derived classes.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by processing.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        void Visualize(Mat image, Mat result, bool printResult = false, bool isRGB = false);

        /// <summary>
        /// Executes processing using previously set inputs.
        /// </summary>
        void Execute();

        /// <summary>
        /// Executes processing using a single input Mat (set to index 0).
        /// </summary>
        /// <param name="input">Input Mat.</param>
        void Execute(Mat input);

        /// <summary>
        /// Executes processing using multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        void Execute(Mat input1, Mat input2);

        /// <summary>
        /// Executes processing using multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="input3">The third input Mat.</param>
        void Execute(Mat input1, Mat input2, Mat input3);

        /// <summary>
        /// Executes processing using multiple input Mats.
        /// </summary>
        /// <param name="inputs">Array of input Mats.</param>
        void Execute(params Mat[] inputs);

        /// <summary>
        /// Executes processing asynchronously with pre-set inputs.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task ExecuteAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Executes processing asynchronously using single input Mat.
        /// </summary>
        /// <param name="input">The input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task ExecuteAsync(Mat input, CancellationToken cancellationToken = default);

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats (optimized for 2 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task ExecuteAsync(Mat input1, Mat input2, CancellationToken cancellationToken = default);

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats (optimized for 3 inputs).
        /// </summary>
        /// <param name="input1">The first input Mat.</param>
        /// <param name="input2">The second input Mat.</param>
        /// <param name="input3">The third input Mat.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task ExecuteAsync(Mat input1, Mat input2, Mat input3, CancellationToken cancellationToken = default);

        /// <summary>
        /// Executes processing asynchronously using multiple input Mats.
        /// </summary>
        /// <param name="inputs">The input Mats.</param>
        /// <param name="cancellationToken">Optional cancellation token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task ExecuteAsync(Mat[] inputs, CancellationToken cancellationToken = default);

        /// <summary>
        /// Cancels the current processing operation if running.
        /// </summary>
        void Cancel();
    }
}
