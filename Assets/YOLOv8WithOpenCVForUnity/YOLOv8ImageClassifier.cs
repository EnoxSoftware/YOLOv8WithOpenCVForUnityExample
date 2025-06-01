#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine;
using YOLOv8WithOpenCVForUnity.UnityIntegration.Worker;
using YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.DataStruct;
using YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.Utils;
using OpenCVRect = OpenCVForUnity.CoreModule.Rect;

namespace YOLOv8WithOpenCVForUnity.Worker
{
    /// <summary>
    /// YOLOv8 image classifier implementation.
    /// This class provides functionality for image classification using the YOLOv8 model implemented with OpenCV's DNN module.
    /// Referring to:
    /// https://github.com/ultralytics/ultralytics/
    ///
    /// [Tested Models]
    /// yolov8n-cls.onnx
    /// yolo11n-cls.onnx
    /// </summary>
    public class YOLOv8ImageClassifier : ProcessingWorkerBase
    {
        protected static readonly Scalar SCALAR_WHITE = new Scalar(255, 255, 255, 255);
        protected static readonly Scalar SCALAR_0 = new Scalar(0, 0, 0, 0);
        protected static readonly Scalar[] SCALAR_PALETTE = new Scalar[]
        {
            new(255, 56, 56, 255),
            new(255, 157, 151, 255),
            new(255, 112, 31, 255),
            new(255, 178, 29, 255),
            new(207, 210, 49, 255),
            new(72, 249, 10, 255),
            new(146, 204, 23, 255),
            new(61, 219, 134, 255),
            new(26, 147, 52, 255),
            new(0, 212, 187, 255),
            new(44, 153, 168, 255),
            new(0, 194, 255, 255),
            new(52, 69, 147, 255),
            new(100, 115, 255, 255),
            new(0, 24, 236, 255),
            new(132, 56, 255, 255),
            new(82, 0, 133, 255),
            new(203, 56, 255, 255),
            new(255, 149, 200, 255),
            new(255, 55, 199, 255)
        };

        protected Size _inputSize;
        protected int _backend;
        protected int _target;
        protected int _numClasses = 1000;
        protected Net _classificationNet;
        protected List<string> _cachedUnconnectedOutLayersNames;
        protected List<string> _classNames;
        protected MatPool _inputSizeMatPool;
        protected Mat _classificationResultBuffer;
        protected Mat _output0Buffer;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
        protected float[] _allResultRowBuffer;
#endif

        /// <summary>
        /// Initializes a new instance of the YOLOv8ImageClassifier class.
        /// </summary>
        /// <param name="modelFilepath">Path to the ONNX model file.</param>
        /// <param name="classesFilepath">Path to the text file containing class names.</param>
        /// <param name="inputSize">Input size for the network (default: 224x224).</param>
        /// <param name="backend">Preferred DNN backend.</param>
        /// <param name="target">Preferred DNN target device.</param>
        public YOLOv8ImageClassifier(string modelFilepath, string classesFilepath, Size inputSize,
                                             int backend = Dnn.DNN_BACKEND_OPENCV, int target = Dnn.DNN_TARGET_CPU)
        {
            if (string.IsNullOrEmpty(modelFilepath))
                throw new ArgumentException("Model filepath cannot be empty.", nameof(modelFilepath));
            if (inputSize == null)
                throw new ArgumentNullException(nameof(inputSize), "Input size cannot be null.");

            _inputSize = new Size(inputSize.width > 0 ? inputSize.width : 224, inputSize.height > 0 ? inputSize.height : 224);
            _backend = backend;
            _target = target;

            try
            {
                _classificationNet = Dnn.readNetFromONNX(modelFilepath);
            }
            catch (Exception e)
            {
                throw new ArgumentException(
                    "Failed to initialize DNN model. Invalid model file path or corrupted model file.", e);
            }
            _classificationNet.setPreferableBackend(_backend);
            _classificationNet.setPreferableTarget(_target);
            _cachedUnconnectedOutLayersNames = _classificationNet.getUnconnectedOutLayersNames();

            _output0Buffer = new Mat();

            if (!string.IsNullOrEmpty(classesFilepath))
            {
                _classNames = ReadClassNames(classesFilepath);
                _numClasses = _classNames.Count;
            }
            else
            {
                _classNames = new List<string>();
            }

            _inputSizeMatPool = new MatPool(_inputSize, CvType.CV_8UC3);
        }

        /// <summary>
        /// Visualizes the best matching classification result on the input image.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by Classify method.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public override void Visualize(Mat image, Mat result, bool printResult = false, bool isRGB = false)
        {
            Visualize(image, result, 0, printResult, isRGB);
        }

        /// <summary>
        /// Visualizes the best matching classification result on the input image.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by Classify method.</param>
        /// <param name="index">Index of the result row to visualize.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public void Visualize(Mat image, Mat result, int index = 0, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();

            if (image != null) image.ThrowIfDisposed();
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            ClassificationData bmData = GetBestMatchData(result, index);
            int classId = bmData.ClassId;
            string label = GetClassLabel(bmData.ClassId) + ", " + bmData.Confidence.ToString("F2");

            var c = SCALAR_PALETTE[classId % SCALAR_PALETTE.Length].ToValueTuple();
            var color = isRGB ? c : (c.v2, c.v1, c.v0, c.v3);

            int[] baseLine = new int[1];
            var labelSize = Imgproc.getTextSizeAsValueTuple(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

            float top = 20f + (float)labelSize.height;
            float left = (float)(image.width() / 2 - labelSize.width / 2f);

            top = Mathf.Max((float)top, (float)labelSize.height);
            Imgproc.rectangle(image, (left, top - labelSize.height),
                (left + labelSize.width, top + baseLine[0]), color, Core.FILLED);
            Imgproc.putText(image, label, (left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_WHITE.ToValueTuple(), 1, Imgproc.LINE_AA);

            if (printResult)
            {
                StringBuilder sb = new StringBuilder(64);
                sb.AppendLine("Best match: " + GetClassLabel(bmData.ClassId) + ", " + bmData.ToString());

                Debug.Log(sb.ToString());
            }
        }

        /// <summary>
        /// Classify the input image.
        /// </summary>
        /// <remarks>
        /// This is a specialized method for image classification that:
        /// - Takes a single BGR image as input
        /// - Returns a Mat containing classification result
        ///
        /// The returned Mat format:
        /// - Single row containing confidence scores for each class
        /// - Columns: [class_0, class_1, class_2, ..., class_n]
        /// - Use ToStructuredData() or GetBestMatchData() to convert to a more convenient format
        ///
        /// Output options:
        /// - useCopyOutput = false (default): Returns a reference to internal buffer (faster but unsafe across executions)
        /// - useCopyOutput = true: Returns a new copy of the result (thread-safe but slightly slower)
        ///
        /// For better performance in async scenarios, use ClassifyAsync instead.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="useCopyOutput">Whether to return a copy of the output (true) or a reference (false).</param>
        /// <returns>A Mat containing classification result. The caller is responsible for disposing this Mat.</returns>
        public virtual Mat Classify(Mat image, bool useCopyOutput = false)
        {
            Execute(image);
            return useCopyOutput ? CopyOutput() : PeekOutput();
        }

        /// <summary>
        /// Classify the input image asynchronously.
        /// </summary>
        /// <remarks>
        /// This is a specialized async method for image classification that:
        /// - Takes a single BGR image as input
        /// - Returns a Mat containing classification result
        ///
        /// The returned Mat format:
        /// - Single row containing confidence scores for each class
        /// - Columns: [class_0, class_1, class_2, ..., class_n]
        /// - Use ToStructuredData() or GetBestMatchData() to convert to a more convenient format.
        ///
        /// Only one classification operation can run at a time.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="cancellationToken">Optional token to cancel the operation.</param>
        /// <returns>A task that represents the asynchronous operation. The task result contains a Mat with classification result. The caller is responsible for disposing this Mat.</returns>
        public virtual async Task<Mat> ClassifyAsync(Mat image, CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(image, cancellationToken);
            return CopyOutput();
        }

        /// <summary>
        /// Classify multiple input images.
        /// </summary>
        /// <remarks>
        /// This is a specialized method for image classification that:
        /// - Takes multiple BGR images as input
        /// - Returns a Mat containing classification results for all images
        ///
        /// The returned Mat format:
        /// - Each row contains confidence scores for each class of one image
        /// - Columns: [class_0, class_1, class_2, ..., class_n]
        /// - Use ToStructuredData() or GetBestMatchData() to convert to a more convenient format
        ///
        /// Output options:
        /// - useCopyOutput = false (default): Returns a reference to internal buffer (faster but unsafe across executions)
        /// - useCopyOutput = true: Returns a new copy of the result (thread-safe but slightly slower)
        ///
        /// For better performance in async scenarios, use ClassifyAsync instead.
        /// </remarks>
        /// <param name="images">Input images in BGR format.</param>
        /// <param name="useCopyOutput">Whether to return a copy of the output (true) or a reference (false).</param>
        /// <returns>A Mat containing classification results for all images. The caller is responsible for disposing this Mat.</returns>
        public virtual Mat Classify(IReadOnlyList<Mat> images, bool useCopyOutput = false)
        {
            Mat[] inputArray = images as Mat[] ?? images.ToArray();
            Execute(inputArray);
            return useCopyOutput ? CopyOutput() : PeekOutput();
        }

        /// <summary>
        /// Classify multiple input images asynchronously.
        /// </summary>
        /// <remarks>
        /// This is a specialized async method for image classification that:
        /// - Takes multiple BGR images as input
        /// - Returns a Mat containing classification results for all images
        ///
        /// The returned Mat format:
        /// - Each row contains confidence scores for each class of one image
        /// - Columns: [class_0, class_1, class_2, ..., class_n]
        /// - Use ToStructuredData() or GetBestMatchData() to convert to a more convenient format.
        ///
        /// Only one classification operation can run at a time.
        /// </remarks>
        /// <param name="images">Input images in BGR format.</param>
        /// <param name="cancellationToken">Optional token to cancel the operation.</param>
        /// <returns>A task that represents the asynchronous operation. The task result contains a Mat with classification results for all images. The caller is responsible for disposing this Mat.</returns>
        public virtual async Task<Mat> ClassifyAsync(IReadOnlyList<Mat> images, CancellationToken cancellationToken = default)
        {
            Mat[] inputArray = images as Mat[] ?? images.ToArray();
            await ExecuteAsync(inputArray, cancellationToken);
            return CopyOutput();
        }

        /// <summary>
        /// Converts the classification result matrix to an array of ClassificationData structures.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <param name="index">Index of the result row to convert.</param>
        /// <returns>Array of ClassificationData structures containing class IDs and confidence scores.</returns>
        public virtual ClassificationData[] ToStructuredData(Mat result, int index = 0)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new ClassificationData[0];
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            if (index < 0 || index >= result.rows())
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");

            int num = result.cols();

            if (_classificationResultBuffer == null)
            {
                _classificationResultBuffer = new Mat(num, 2, CvType.CV_32FC1);
                float[] arange = Enumerable.Range(0, num).Select(i => (float)i).ToArray();
                using (Mat result_Nx1_col_1 = _classificationResultBuffer.col(1))
                {
                    result_Nx1_col_1.put(0, 0, arange);
                }
            }

            using (Mat result_row = result.row(index))
            using (Mat result_Nx1 = result_row.reshape(1, num))
            using (Mat result_Nx1_col_0 = _classificationResultBuffer.col(0))
            {
                result_Nx1.copyTo(result_Nx1_col_0);
            }

            var dst = new ClassificationData[num];
            MatUtils.copyFromMat(_classificationResultBuffer, dst);

            return dst;
        }

        /// <summary>
        /// Gets the top K sorted classification result.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <param name="topK">Number of top result to return.</param>
        /// <param name="index">Index of the result row to get sorted data from.</param>
        /// <returns>Array of sorted ClassificationData structures containing top K result.</returns>
        public virtual ClassificationData[] GetSortedData(Mat result, int topK = 5, int index = 0)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new ClassificationData[0];
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            if (index < 0 || index >= result.rows())
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");

            int num = result.cols();

            if (topK < 1 || topK > num) topK = num;

            // Get raw data
            var data = ToStructuredData(result, index);
            if (data.Length == 0)
                return data;

            // If we need all elements, just sort the entire array
            if (topK == num)
            {
                Array.Sort(data, (a, b) => b.Confidence.CompareTo(a.Confidence));
                return data;
            }

            // Otherwise, use partial sort to get top K elements
            var sortedData = new ClassificationData[topK];
            var indices = new int[data.Length];
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;

            // Partial sort indices based on confidence values
            for (int i = 0; i < topK; i++)
            {
                int maxIndex = i;
                float maxConfidence = data[indices[i]].Confidence;

                for (int j = i + 1; j < indices.Length; j++)
                {
                    if (data[indices[j]].Confidence > maxConfidence)
                    {
                        maxIndex = j;
                        maxConfidence = data[indices[j]].Confidence;
                    }
                }

                if (maxIndex != i)
                {
                    int temp = indices[i];
                    indices[i] = indices[maxIndex];
                    indices[maxIndex] = temp;
                }

                sortedData[i] = data[indices[i]];
            }

            return sortedData;
        }

        /// <summary>
        /// Gets the best matching classification result.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <param name="index">Index of the result row to get the best match from.</param>
        /// <returns>ClassificationData structure containing the best match.</returns>
        public virtual ClassificationData GetBestMatchData(Mat result, int index = 0)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new ClassificationData();
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            if (index < 0 || index >= result.rows())
                throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range.");

            float maxVal = float.MinValue;
            int maxLoc = 0;

            Span<float> data;
            using (Mat result_row = result.row(index))
            {
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                data = result_row.AsSpan<float>();
#else
                int requiredResultRowLen = (int)result_row.total();
                if (_allResultRowBuffer == null || _allResultRowBuffer.Length < requiredResultRowLen)
                    _allResultRowBuffer = new float[requiredResultRowLen];
                result_row.get(0, 0, _allResultRowBuffer);
                data = _allResultRowBuffer.AsSpan(0, requiredResultRowLen);
#endif
            }

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] > maxVal)
                {
                    maxVal = data[i];
                    maxLoc = i;
                }
            }

            return new ClassificationData(maxVal, maxLoc);
        }

        /// <summary>
        /// Converts the classification result matrix to an array of ClassificationData structures for all rows.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <returns>List of arrays of ClassificationData structures containing class IDs and confidence scores.</returns>
        public virtual List<ClassificationData[]> ToStructuredDatas(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new List<ClassificationData[]>();
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            var datas = new List<ClassificationData[]>();
            for (int i = 0; i < result.rows(); i++)
            {
                datas.Add(ToStructuredData(result, i));
            }
            return datas;
        }

        /// <summary>
        /// Gets the top K sorted classification results for all rows.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <param name="topK">Number of top results to return.</param>
        /// <returns>List of arrays of sorted ClassificationData structures containing top K results.</returns>
        public virtual List<ClassificationData[]> GetSortedDatas(Mat result, int topK = 5)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new List<ClassificationData[]>();
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            var datas = new List<ClassificationData[]>();
            for (int i = 0; i < result.rows(); i++)
            {
                datas.Add(GetSortedData(result, topK, i));
            }
            return datas;
        }

        /// <summary>
        /// Gets the best matching classification results for all rows.
        /// </summary>
        /// <param name="result">Classification result matrix from Execute method.</param>
        /// <returns>Array of ClassificationData structures containing the best matches.</returns>
        public virtual ClassificationData[] GetBestMatchDatas(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new ClassificationData[0];
            if (result.cols() < _numClasses)
                throw new ArgumentException("Invalid result matrix. It must have at least " + _numClasses + " columns.");

            var datas = new ClassificationData[result.rows()];
            for (int i = 0; i < result.rows(); i++)
            {
                datas[i] = GetBestMatchData(result, i);
            }
            return datas;
        }

        /// <summary>
        /// Gets the class label for the given class ID.
        /// </summary>
        /// <param name="id">Class ID.</param>
        /// <returns>String representation of the class label.</returns>
        public virtual string GetClassLabel(float id)
        {
            ThrowIfDisposed();

            return ClassLabelUtils.GetClassLabel(id, _classNames);
        }

        /// <summary>
        /// Gets all class labels.
        /// </summary>
        /// <returns>Array of class label strings.</returns>
        public virtual string[] GetClassLabels()
        {
            ThrowIfDisposed();

            return _classNames.ToArray();
        }

        protected override Mat[] RunCoreProcessing(Mat[] inputs)
        {
            ThrowIfDisposed();

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs), "Inputs cannot be null.");

            for (int i = 0; i < inputs.Length; i++)
            {
                Mat input = inputs[i];
                if (input == null)
                    throw new ArgumentNullException(nameof(inputs), "inputs[" + i + "] cannot be null.");

                if (input != null) input.ThrowIfDisposed();
                if (input.channels() != 3)
                    throw new ArgumentException("The input image must be in BGR format. inputs[" + i + "]");
            }

            // Preprocess
            Mat inputBlob = PreProcess(inputs);

            // Forward
            _classificationNet.setInput(inputBlob);
            List<Mat> outputBlobs = new List<Mat>();
            _classificationNet.forward(outputBlobs, _cachedUnconnectedOutLayersNames);

            // Postprocess
            Mat outputBlob = PostProcess(outputBlobs[0]);

            // Any rewriting of buffer data must be done within the lock statement
            // Do not return the buffer itself because it will be destroyed,
            // but return a submat of the same size as the result extracted using rowRange
            Mat submat = null;
            lock (_lockObject)
            {
                // Check if _output0Buffer needs to be resized
                if (_output0Buffer == null || _output0Buffer.rows() < outputBlob.rows() || _output0Buffer.cols() < outputBlob.cols())
                {
                    _output0Buffer.create(outputBlob.rows(), outputBlob.cols(), outputBlob.type());
                }

                // If buffer is larger, use rowRange to copy only the needed portion
                submat = _output0Buffer.rowRange(0, outputBlob.rows());
                outputBlob.copyTo(submat);
            }

            inputBlob.Dispose();
            for (int i = 0; i < outputBlobs.Count; i++)
            {
                outputBlobs[i].Dispose();
            }

            return new Mat[] { submat }; // [n, num_classes]
        }

        protected override Task<Mat[]> RunCoreProcessingAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(RunCoreProcessing(inputs));
        }

        protected virtual Mat PreProcess(IReadOnlyList<Mat> images)
        {
            // https://github.com/ultralytics/ultralytics/blob/d74a5a9499acf1afd13d970645e5b1cfcadf4a8f/ultralytics/data/augment.py#L1059

            List<Mat> inputSizeMats = new List<Mat>();
            foreach (var image in images)
            {
                Mat inputSizeMat = _inputSizeMatPool.Get();

                // Resizes and crops the center of the image to a square shape
                int imh = image.height();
                int imw = image.width();
                int m = Mathf.Min(imh, imw);
                int top = (int)((imh - m) / 2f);
                int left = (int)((imw - m) / 2f);
                using (Mat image_crop = new Mat(image, new OpenCVRect(0, 0, image.width(), image.height()).intersect(new OpenCVRect(left, top, m, m))))
                {
                    Imgproc.resize(image_crop, inputSizeMat, _inputSize);
                }
                inputSizeMats.Add(inputSizeMat);
            }

            // Create a 4D blob from a frame.
            // YOLOv8 Image Classifier is scalefactor = 1.0 / 255.0, swapRB = true
            Mat blob = Dnn.blobFromImages(inputSizeMats, 1.0 / 255.0, _inputSize, SCALAR_0, true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

            foreach (var inputSizeMat in inputSizeMats)
            {
                _inputSizeMatPool.Return(inputSizeMat);
            }

            return blob; // [n, 3, h, w]
        }

        protected virtual Mat PostProcess(Mat outputBlob)
        {
            Mat outputBlob_0 = outputBlob;

            if (outputBlob_0.cols() != _numClasses)
            {
                Debug.LogWarning("The number of classes and output shapes are different. " +
                                "( outputBlob_0.cols():" + outputBlob_0.cols() + " != numClasses:" + _numClasses + " )\n" +
                                "When using a custom model, be sure to set the correct number of classes by loading the appropriate custom classesFile.");

                _numClasses = outputBlob_0.cols();
            }

            return outputBlob_0; // [n, num_classes]
        }

        protected virtual List<string> ReadClassNames(string filename)
        {
            return ClassLabelUtils.ReadClassNames(filename);
        }

        protected override void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                _classificationNet?.Dispose(); _classificationNet = null;
                _inputSizeMatPool?.Dispose(); _inputSizeMatPool = null;
                _classificationResultBuffer?.Dispose(); _classificationResultBuffer = null;
                _output0Buffer?.Dispose(); _output0Buffer = null;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
                _allResultRowBuffer = null;
#endif
            }

            base.Dispose(disposing);
        }
    }
}
#endif
