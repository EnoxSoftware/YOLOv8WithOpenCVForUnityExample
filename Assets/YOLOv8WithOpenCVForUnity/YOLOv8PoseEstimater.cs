#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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

namespace YOLOv8WithOpenCVForUnity.Worker
{
    /// <summary>
    /// YOLOv8 pose estimator implementation.
    /// This class provides functionality for pose estimation using the YOLOv8 model implemented with OpenCV's DNN module.
    /// Referring to:
    /// https://github.com/ultralytics/ultralytics/
    ///
    /// [Tested Models]
    /// yolov8n-pose.onnx
    /// yolov11n-pose.onnx
    /// </summary>
    public class YOLOv8PoseEstimater : ProcessingWorkerBase
    {
        public enum NMSStrategy
        {
            /// <summary>
            /// Performs NMS ignoring class information (class-agnostic).
            /// </summary>
            ClassAgnostic,

            /// <summary>
            /// Performs NMS with OpenCV's NMSBoxesBatched, where multi-class boxes may still suppress each other
            /// if their IoU is high, as suppression is not strictly class-wise.
            /// </summary>
            OpenCVNMSBoxesBatched,

            /// <summary>
            /// Performs NMS separately for each class (Ultralytics YOLO style).
            /// </summary>
            ClassWise
        }

        protected static readonly Scalar SCALAR_WHITE = new Scalar(255, 255, 255, 255);
        protected static readonly Scalar SCALAR_114 = new Scalar(114, 114, 114, 114);
        protected static readonly Scalar SCALAR_0 = new Scalar(0, 0, 0, 0);
        protected static readonly Scalar SCALAR_1 = new Scalar(1, 1, 1, 1);
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
        protected static readonly Scalar[] POSE_PALETTE = new Scalar[] {
            new(0, 128, 255, 255),
            new(51, 153, 255, 255),
            new(102, 178, 255, 255),
            new(0, 230, 230, 255),
            new(255, 153, 255, 255),
            new(255, 204, 153, 255),
            new(255, 102, 255, 255),
            new(255, 51, 255, 255),
            new(255, 178, 102, 255),
            new(255, 153, 51, 255),
            new(153, 153, 255, 255),
            new(102, 102, 255, 255),
            new(51, 51, 255, 255),
            new(153, 255, 153, 255),
            new(102, 255, 102, 255),
            new(51, 255, 51, 255),
            new(0, 255, 0, 255),
            new(255, 0, 0, 255),
            new(0, 0, 255, 255),
            new(255, 255, 255, 255)
        };
        protected static readonly int[,] SKELETON = new int[,] { { 16, 14 }, { 14, 12 }, { 17, 15 }, { 15, 13 }, { 12, 13 },
                                                                 { 6, 12 }, { 7, 13 }, { 6, 7 }, { 6, 8 }, {7, 9 },
                                                                 { 8, 10 }, {9, 11 }, {2, 3 }, {1, 2 }, {1, 3 },
                                                                 {2, 4 }, {3, 5 }, {4, 6 }, {5, 7 } };
        protected static readonly int[] LIMB_COLOR_INDEX = new int[] { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
        protected static readonly int[] KPT_COLOR_INDEX = new int[] { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
        protected static readonly int BOUNDING_BOX_COLUMNS = 4;
        protected static readonly int DETECTION_RESULT_COLUMNS = 6;
        protected static readonly int DETECTION_BOX_START_INDEX = 0;
        protected static readonly int DETECTION_CONFIDENCE_INDEX = 4;
        protected static readonly int DETECTION_CLASS_ID_INDEX = 5;
        protected static readonly int KPTS_COLUMNS = 51; // 17*3

        protected Size _inputSize;
        protected float _confThreshold;
        protected float _nmsThreshold;
        protected int _topK;
        protected int _backend;
        protected int _target;
        protected int _numClasses = 1;
        protected Net _poseEstimationNet;
        protected List<string> _cachedUnconnectedOutLayersNames;
        protected List<string> _classNames;
        protected Mat _paddedImg;
        protected Mat _preNMS_Nx56;
        protected Mat _preNMS_box_xywh;
        protected MatOfRect2d _NMS_boxes;
        protected MatOfFloat _NMS_confidences;
        protected MatOfInt _NMS_classIds;
        protected Mat _output0Buffer;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
        protected float[] _allBoxClsconfsKptsBuffer;
        protected float[] _allPreNMSBuffer;
        protected int[] _allIndicesBuffer;
        protected float[] _allResultBuffer;
#endif

        /// <summary>
        /// Gets or sets the NMS strategy to use for pose estimation.
        /// </summary>
        public NMSStrategy SelectedNMSStrategy { get; set; } = NMSStrategy.ClassWise;

        /// <summary>
        /// Initializes a new instance of the YOLOv8PoseEstimater class.
        /// </summary>
        /// <param name="modelFilepath">Path to the ONNX model file.</param>
        /// <param name="classesFilepath">Path to the text file containing class names.</param>
        /// <param name="inputSize">Input size for the network (default: 640x640).</param>
        /// <param name="confThreshold">Confidence threshold for filtering detections.</param>
        /// <param name="nmsThreshold">Non-maximum suppression threshold.</param>
        /// <param name="topK">Maximum number of output detections.</param>
        /// <param name="backend">Preferred DNN backend.</param>
        /// <param name="target">Preferred DNN target device.</param>
        public YOLOv8PoseEstimater(string modelFilepath, string classesFilepath, Size inputSize,
                                             float confThreshold = 0.25f, float nmsThreshold = 0.45f, int topK = 300,
                                             int backend = Dnn.DNN_BACKEND_OPENCV, int target = Dnn.DNN_TARGET_CPU)
        {
            if (string.IsNullOrEmpty(modelFilepath))
                throw new ArgumentException("Model filepath cannot be empty.", nameof(modelFilepath));
            if (inputSize == null)
                throw new ArgumentNullException(nameof(inputSize), "Input size cannot be null.");

            _inputSize = new Size(inputSize.width > 0 ? inputSize.width : 640, inputSize.height > 0 ? inputSize.height : 640);
            _confThreshold = Mathf.Clamp01(confThreshold);
            _nmsThreshold = Mathf.Clamp01(nmsThreshold);
            _topK = Math.Max(1, topK);
            _backend = backend;
            _target = target;

            try
            {
                _poseEstimationNet = Dnn.readNetFromONNX(modelFilepath);
            }
            catch (Exception e)
            {
                throw new ArgumentException(
                    "Failed to initialize DNN model. Invalid model file path or corrupted model file.", e);
            }
            _poseEstimationNet.setPreferableBackend(_backend);
            _poseEstimationNet.setPreferableTarget(_target);
            _cachedUnconnectedOutLayersNames = _poseEstimationNet.getUnconnectedOutLayersNames();

            _output0Buffer = new Mat(_topK, DETECTION_RESULT_COLUMNS + KPTS_COLUMNS, CvType.CV_32FC1);

            if (!string.IsNullOrEmpty(classesFilepath))
            {
                _classNames = ReadClassNames(classesFilepath);
                _numClasses = _classNames.Count;
            }
            else
            {
                _classNames = new List<string>();
            }
        }

        /// <summary>
        /// Visualizes pose estimation result on the input image.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by Estimate method.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public override void Visualize(Mat image, Mat result, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();

            if (image != null) image.ThrowIfDisposed();
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.cols() < DETECTION_RESULT_COLUMNS + KPTS_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 56 columns.");

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<PoseEstimationCOCO17DataUnsafe> data = ToStructuredDataAsSpanUnsafe(result);
#else
            PoseEstimationCOCO17Data[] data = ToStructuredData(result);
#endif

            foreach (var d in data)
            {
                float left = d.X1;
                float top = d.Y1;
                float right = d.X2;
                float bottom = d.Y2;
                float conf = d.Confidence;
                int classId = d.ClassId;

                var c = SCALAR_PALETTE[classId % SCALAR_PALETTE.Length].ToValueTuple();
                var color = isRGB ? c : (c.v2, c.v1, c.v0, c.v3);

                Imgproc.rectangle(image, (left, top), (right, bottom), color, 2);

                string label = $"{GetClassLabel(classId)}, {conf:F2}";

                int[] baseLine = new int[1];
                var labelSize = Imgproc.getTextSizeAsValueTuple(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

                top = Mathf.Max((float)top, (float)labelSize.height);
                Imgproc.rectangle(image, (left, top - labelSize.height),
                    (left + labelSize.width, top + baseLine[0]), color, Core.FILLED);
                Imgproc.putText(image, label, (left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_WHITE.ToValueTuple(), 1, Imgproc.LINE_AA);

                // draw points
                int radius = 5;
                var landmarks = d.GetLandmarks();
                for (int i = 0; i < landmarks.Length; i++)
                {
                    var point_c = POSE_PALETTE[KPT_COLOR_INDEX[i]].ToValueTuple();
                    var color_k = isRGB ? point_c : (point_c.v2, point_c.v1, point_c.v0, point_c.v3);

                    Vec3f landmark = landmarks[i];
                    float x_coord = landmark.Item1;
                    float y_coord = landmark.Item2;

                    if (x_coord % image.width() != 0 && y_coord % image.height() != 0)
                    {
                        float point_conf = landmarks[i].Item3;
                        if (point_conf < 0.5)
                            continue;

                        Imgproc.circle(image, (x_coord, y_coord), radius, color_k, -1, Imgproc.LINE_AA);
                    }
                }

                // draw lines
                for (int i = 0; i < SKELETON.GetLength(0); i++)
                {
                    Vec3f landmark1 = landmarks[SKELETON[i, 0] - 1];
                    float pos1_x = landmark1.Item1;
                    float pos1_y = landmark1.Item2;
                    float conf1 = landmark1.Item3;
                    Vec3f landmark2 = landmarks[SKELETON[i, 1] - 1];
                    float pos2_x = landmark2.Item1;
                    float pos2_y = landmark2.Item2;
                    float conf2 = landmark2.Item3;

                    if (conf1 < 0.5f || conf2 < 0.5f)
                        continue;

                    if (pos1_x % image.width() == 0 || pos1_y % image.height() == 0 || pos1_x < 0 || pos1_y < 0)
                        continue;

                    if (pos2_x % image.width() == 0 || pos2_y % image.height() == 0 || pos2_x < 0 || pos2_y < 0)
                        continue;

                    var line_c = POSE_PALETTE[LIMB_COLOR_INDEX[i]].ToValueTuple();
                    var color_k = isRGB ? line_c : (line_c.v2, line_c.v1, line_c.v0, line_c.v3);

                    Imgproc.line(image, (pos1_x, pos1_y), (pos2_x, pos2_y), color_k, 2, Imgproc.LINE_AA);
                }
            }

            if (printResult)
            {
                StringBuilder sb = new StringBuilder(512);

                for (int i = 0; i < data.Length; ++i)
                {
                    var d = data[i];
                    sb.AppendFormat("-----------object {0}-----------", i + 1);
                    sb.AppendLine();
                    sb.Append("Class: ").Append(GetClassLabel(d.ClassId));
                    sb.AppendLine();
                    sb.AppendFormat("Confidence: {0:F4}", d.Confidence);
                    sb.AppendLine();
                    sb.AppendFormat("Box: {0:F0} {1:F0} {2:F0} {3:F0}", d.X1, d.Y1, d.X2, d.Y2);
                    sb.AppendLine();
                    ReadOnlySpan<Vec3f> landmarks = d.GetLandmarks();
                    sb.Append("Landmarks:");
                    foreach (var p in landmarks)
                    {
                        sb.Append(" ");
                        sb.Append(p.ToString());
                    }
                    sb.AppendLine();
                }

                Debug.Log(sb.ToString());
            }
        }

        /// <summary>
        /// Estimates poses in the input image.
        /// </summary>
        /// <remarks>
        /// This is a specialized method for pose estimation that:
        /// - Takes a single BGR image as input
        /// - Returns a Mat containing pose estimation result (6 columns per pose)
        ///
        /// The returned pose estimation result Mat format:
        /// - Each row represents one pose
        /// - Columns: [x1, y1, x2, y2, confidence, classId, landmark1_x, landmark1_y, landmark1_conf, ... landmark17_x, landmark17_y, landmark17_conf]
        /// - Use ToStructuredDataUnsafe() or ToStructuredDataAsSpanUnsafe() to convert to a more convenient format
        ///
        /// Output options:
        /// - useCopyOutput = false (default): Returns a reference to internal buffer (faster but unsafe across executions)
        /// - useCopyOutput = true: Returns a new copy of the result (thread-safe but slightly slower)
        ///
        /// For better performance in async scenarios, use EstimateAsync instead.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="useCopyOutput">Whether to return a copy of the output (true) or a reference (false).</param>
        /// <returns>A Mat containing pose estimation result. The caller is responsible for disposing this Mat.</returns>
        public virtual Mat Estimate(Mat image, bool useCopyOutput = false)
        {
            Execute(image);
            return useCopyOutput ? CopyOutput(0) : PeekOutput(0);
        }

        /// <summary>
        /// Estimates poses in the input image asynchronously.
        /// </summary>
        /// <remarks>
        /// This is a specialized async method for pose estimation that:
        /// - Takes a single BGR image as input
        /// - Returns a Mat containing pose estimation result (6 columns per pose)
        ///
        /// The returned pose estimation result Mat format:
        /// - Each row represents one pose
        /// - Columns: [x1, y1, x2, y2, confidence, classId, landmark1_x, landmark1_y, landmark1_conf, ... landmark17_x, landmark17_y, landmark17_conf]
        /// - Use ToStructuredDataUnsafe() or ToStructuredDataAsSpanUnsafe() to convert to a more convenient format
        ///
        /// Only one pose estimation operation can run at a time.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="cancellationToken">Optional token to cancel the operation.</param>
        /// <returns>A task that represents the asynchronous operation. The task result contains a Mat containing pose estimation result. The caller is responsible for disposing this Mat.</returns>
        public virtual async Task<Mat> EstimateAsync(Mat image, CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(image, cancellationToken);
            return CopyOutput(0);
        }

        /// <summary>
        /// Converts the pose estimation result matrix to an array of PoseEstimationCOCO17Data structures.
        /// </summary>
        /// <param name="result">Pose estimation result matrix from Execute method.</param>
        /// <returns>Array of PoseEstimationCOCO17Data structures containing pose estimation information.</returns>
        public virtual PoseEstimationCOCO17Data[] ToStructuredData(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new PoseEstimationCOCO17Data[0];
            if (result.cols() < DETECTION_RESULT_COLUMNS + KPTS_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 56 columns.");

            var dst = new PoseEstimationCOCO17Data[result.rows()];
            for (int i = 0; i < result.rows(); i++)
            {
                using (Mat row = result.row(i))
                {
                    dst[i] = Marshal.PtrToStructure<PoseEstimationCOCO17Data>((IntPtr)row.dataAddr());
                }
            }

            return dst;
        }

#if !OPENCV_DONT_USE_UNSAFE_CODE
        /// <summary>
        /// Converts the pose estimation result matrix to an array of PoseEstimationCOCO17DataUnsafe structures.
        /// </summary>
        /// <param name="result">Pose estimation result matrix from Execute method.</param>
        /// <returns>Array of PoseEstimationCOCO17DataUnsafe structures containing pose estimation information.</returns>
        public virtual PoseEstimationCOCO17DataUnsafe[] ToStructuredDataUnsafe(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new PoseEstimationCOCO17DataUnsafe[0];
            if (result.cols() < DETECTION_RESULT_COLUMNS + KPTS_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 56 columns.");

            var dst = new PoseEstimationCOCO17DataUnsafe[result.rows()];
            MatUtils.copyFromMat(result, dst);

            return dst;
        }
#endif

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
        /// <summary>
        /// Converts the pose estimation result matrix to a span of PoseEstimationCOCO17DataUnsafe structures.
        /// </summary>
        /// <param name="result">Pose estimation result matrix from Execute method.</param>
        /// <returns>Span of PoseEstimationCOCO17DataUnsafe structures containing pose estimation information.</returns>
        public virtual Span<PoseEstimationCOCO17DataUnsafe> ToStructuredDataAsSpanUnsafe(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return Span<PoseEstimationCOCO17DataUnsafe>.Empty;
            if (result.cols() < DETECTION_RESULT_COLUMNS + KPTS_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 56 columns.");
            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return result.AsSpan<PoseEstimationCOCO17DataUnsafe>();
        }
#endif

        /// <summary>
        /// Gets the class label for the given class ID.
        /// </summary>
        /// <param name="id">Class ID.</param>
        /// <returns>Class label string. Returns the ID as string if no label is found.</returns>
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

            if (inputs == null || inputs.Length < 1)
                throw new ArgumentNullException(nameof(inputs), "Inputs cannot be null or have less than 1 elements.");

            if (inputs[0] == null)
                throw new ArgumentNullException(nameof(inputs), "inputs[0] cannot be null.");

            Mat image = inputs[0];

            if (image != null) image.ThrowIfDisposed();
            if (image.channels() != 3)
                throw new ArgumentException("The input image must be in BGR format.");

            // Preprocess
            Mat inputBlob = PreProcess(image);

            // Forward
            _poseEstimationNet.setInput(inputBlob);
            List<Mat> outputBlobs = new List<Mat>();
            try
            {
                _poseEstimationNet.forward(outputBlobs, _cachedUnconnectedOutLayersNames);
            }
            catch (Exception e)
            {
                inputBlob.Dispose();
                throw new ArgumentException(
                    "The input size specified in the constructor may not match the model's expected input size. " +
                    "Please verify the correct input size for your model and update the constructor parameters accordingly.", e);
            }

            // Postprocess
            Mat submat = PostProcess(outputBlobs, image.sizeAsValueTuple()); // submat of _output0Buffer is returned

            // Any rewriting of buffer data must be done within the lock statement
            // Do not return the buffer itself because it will be destroyed,
            // but return a submat of the same size as the result extracted using rowRange

            inputBlob.Dispose();
            for (int i = 0; i < outputBlobs.Count; i++)
            {
                outputBlobs[i].Dispose();
            }

            return new Mat[] { submat }; // [n, 6 + 51] (xyxy, conf, cls, kpts)
        }

        protected override Task<Mat[]> RunCoreProcessingAsync(Mat[] inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(RunCoreProcessing(inputs));
        }

        protected virtual Mat PreProcess(Mat image)
        {
            // https://github.com/ultralytics/ultralytics/blob/d74a5a9499acf1afd13d970645e5b1cfcadf4a8f/ultralytics/data/augment.py#L645

            // Add padding to make it input size.
            // (padding to center the image)
            float ratio = Mathf.Max((float)image.cols() / (float)_inputSize.width, (float)image.rows() / (float)_inputSize.height);
            int paddedImgW = (int)Mathf.Ceil((float)_inputSize.width * ratio);
            int paddedImgH = (int)Mathf.Ceil((float)_inputSize.height * ratio);

            if (_paddedImg == null || _paddedImg.width() != paddedImgW || _paddedImg.height() != paddedImgH)
            {
                if (_paddedImg == null)
                    _paddedImg = new Mat();
                _paddedImg.create(paddedImgH, paddedImgW, image.type());
                _paddedImg.setTo(SCALAR_114);
            }

            using (Mat _paddedImg_roi = new Mat(_paddedImg,
                ((_paddedImg.cols() - image.cols()) / 2, (_paddedImg.rows() - image.rows()) / 2, image.cols(), image.rows())))
            {
                image.copyTo(_paddedImg_roi);
            }

            // Create a 4D blob from a frame.
            // YOLOv8 is scalefactor = 1.0 / 255.0, swapRB = true
            Mat blob = Dnn.blobFromImage(_paddedImg, 1.0 / 255.0, _inputSize, SCALAR_0, true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

            return blob; // [1, 3, h, w]
        }

        protected virtual Mat PostProcess(List<Mat> outputBlobs, (double width, double height) originalSize)
        {
            Mat outputBlob_0 = outputBlobs[0]; // 1 * (4 + _numClasses + 51) * N

            if (outputBlob_0.size(1) != 4 + _numClasses + KPTS_COLUMNS)
            {
                Debug.LogWarning("The number of classes and output shapes are different. " +
                                "( outputBlob_0.size(1):" + outputBlob_0.size(1) + " != 4 + numClasses:" + _numClasses + " + KPTS_COLUMNS:" + KPTS_COLUMNS + " )\n" +
                                "When using a custom model, be sure to set the correct number of classes by loading the appropriate custom classesFile.");

                _numClasses = outputBlob_0.size(1) - 4 - KPTS_COLUMNS;
            }

            int num = outputBlob_0.size(1);
            using (Mat outputBlob_NxBoxConfClsconfsKpts = outputBlob_0.reshape(1, num))
            {
                // pre-NMS
                PreNMS_BoxClsconfsKpts_CXCYWHtoX1Y1X2Y2_YOLOv8(outputBlob_NxBoxConfClsconfsKpts, ref _preNMS_Nx56, _confThreshold, _topK);
            }

            Mat result = null;
            using (MatOfInt indices = new MatOfInt())
            {
                // NMS
                NMS(_preNMS_Nx56, _confThreshold, _nmsThreshold, indices, 1f, _topK, SelectedNMSStrategy);

                lock (_lockObject)
                {
                    // Create result
                    result = CreateResultFromBuffer_Nx56(_preNMS_Nx56, indices, _output0Buffer);

                    // Scale boxes and landmarks
                    ScaleBoxesAndKpts(result, _inputSize.ToValueTuple(), originalSize);
                }
            }

            // [
            //   [xyxy, conf, cls, kpts]
            //   ...
            //   [xyxy, conf, cls, kpts]
            // ]
            return result;
        }

        protected virtual void PreNMS_BoxClsconfsKpts_CXCYWHtoX1Y1X2Y2_YOLOv8(Mat boxClsconfsKpts, ref Mat preNMS_Nx56, float score_threshold, int top_k = 300)
        {
            if (!boxClsconfsKpts.isContinuous())
                throw new ArgumentException("boxClsconfsKpts is not continuous.");

            const int CLASS_CONFIDENCES_INDEX = 4;

            int boxClsconfsKpts_rows = boxClsconfsKpts.rows();
            int boxClsconfsKpts_cols = boxClsconfsKpts.cols();

            if (preNMS_Nx56 == null || preNMS_Nx56.rows() < top_k)
            {
                if (preNMS_Nx56 == null)
                    preNMS_Nx56 = new Mat();
                preNMS_Nx56.create(top_k, DETECTION_RESULT_COLUMNS + KPTS_COLUMNS, CvType.CV_32FC1);
                preNMS_Nx56.setTo(SCALAR_0);
            }
            int num_preNMS = preNMS_Nx56.rows();

            // Initialize output data (confidences only)
            using (Mat preNMS_Nx56_col_4 = preNMS_Nx56.col(DETECTION_CONFIDENCE_INDEX))
            {
                preNMS_Nx56_col_4.setTo(SCALAR_0);
            }

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<float> allBoxClsconfsKpts = boxClsconfsKpts.AsSpan<float>();
            Span<float> allPreNMS = preNMS_Nx56.AsSpan<float>();
#else
            int requiredBoxClsconfsKptsLen = boxClsconfsKpts_rows * boxClsconfsKpts_cols;
            int requiredPreNMSLen = num_preNMS * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);

            if (_allBoxClsconfsKptsBuffer == null || _allBoxClsconfsKptsBuffer.Length < requiredBoxClsconfsKptsLen)
                _allBoxClsconfsKptsBuffer = new float[requiredBoxClsconfsKptsLen];
            if (_allPreNMSBuffer == null || _allPreNMSBuffer.Length < requiredPreNMSLen)
                _allPreNMSBuffer = new float[requiredPreNMSLen];

            boxClsconfsKpts.get(0, 0, _allBoxClsconfsKptsBuffer);
            preNMS_Nx56.get(0, 0, _allPreNMSBuffer);
            ReadOnlySpan<float> allBoxClsconfsKpts = _allBoxClsconfsKptsBuffer.AsSpan(0, requiredBoxClsconfsKptsLen);
            Span<float> allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
#endif

            int ind = 0;
            for (int i = 0; i < boxClsconfsKpts_cols; ++i)
            {
                float maxVal = float.MinValue;
                int maxIdx = -1;
                for (int j = 0; j < _numClasses; ++j)
                {
                    float conf = allBoxClsconfsKpts[(CLASS_CONFIDENCES_INDEX + j) * boxClsconfsKpts_cols + i];
                    if (conf > maxVal)
                    {
                        maxVal = conf;
                        maxIdx = j;
                    }
                }

                if (maxVal > score_threshold)
                {
                    // If we've reached the capacity of our pre-NMS buffer, double its size to accommodate more detections.
                    if (ind >= num_preNMS)
                    {
                        Mat new_preNMS_Nx56 = new Mat(num_preNMS * 2, DETECTION_RESULT_COLUMNS + KPTS_COLUMNS, CvType.CV_32FC1, SCALAR_0.ToValueTuple());
                        using (Mat new_preNMS_Nx56_roi = new_preNMS_Nx56.rowRange(0, num_preNMS))
                        {
                            preNMS_Nx56.copyTo(new_preNMS_Nx56_roi);
                        }
                        preNMS_Nx56.Dispose();
                        preNMS_Nx56 = new_preNMS_Nx56;
                        num_preNMS = preNMS_Nx56.rows();

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                        allPreNMS = preNMS_Nx56.AsSpan<float>();
#else
                        requiredPreNMSLen = num_preNMS * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
                        float[] newBuffer = new float[requiredPreNMSLen];
                        Array.Copy(_allPreNMSBuffer, newBuffer, _allPreNMSBuffer.Length);
                        _allPreNMSBuffer = newBuffer;
                        allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
#endif
                    }

                    int preNMSIdx = ind * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);

                    // Convert from [cx,cy,w,h] to [x1,y1,x2,y2]
                    float cx = allBoxClsconfsKpts[0 * boxClsconfsKpts_cols + i];
                    float cy = allBoxClsconfsKpts[1 * boxClsconfsKpts_cols + i];
                    float w = allBoxClsconfsKpts[2 * boxClsconfsKpts_cols + i];
                    float h = allBoxClsconfsKpts[3 * boxClsconfsKpts_cols + i];

                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;
                    float x2 = cx + w / 2;
                    float y2 = cy + h / 2;

                    allPreNMS[preNMSIdx] = x1;
                    allPreNMS[preNMSIdx + 1] = y1;
                    allPreNMS[preNMSIdx + 2] = x2;
                    allPreNMS[preNMSIdx + 3] = y2;
                    allPreNMS[preNMSIdx + DETECTION_CONFIDENCE_INDEX] = maxVal;
                    allPreNMS[preNMSIdx + DETECTION_CLASS_ID_INDEX] = maxIdx;

                    // Copy kpts data
                    int kptsIdx = (CLASS_CONFIDENCES_INDEX + _numClasses) * boxClsconfsKpts_cols + i;
                    for (int k = 0; k < KPTS_COLUMNS; ++k)
                    {
                        allPreNMS[preNMSIdx + DETECTION_RESULT_COLUMNS + k] = allBoxClsconfsKpts[kptsIdx + k * boxClsconfsKpts_cols];
                    }

                    ind++;
                }
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            preNMS_Nx56.put(0, 0, _allPreNMSBuffer);
#endif
        }

        protected virtual void NMS(Mat preNMS_Nx6, float score_threshold, float nms_threshold, MatOfInt indices,
                                     float eta = 1f, int top_k = 300, NMSStrategy nmsStrategy = NMSStrategy.ClassWise)
        {
            if (indices == null)
                throw new ArgumentNullException("indices");

            int num_preNMS = preNMS_Nx6.rows();
            using (Mat preNMS_box = preNMS_Nx6.colRange((DETECTION_BOX_START_INDEX, BOUNDING_BOX_COLUMNS)))
            using (Mat preNMS_confidence = preNMS_Nx6.colRange((DETECTION_CONFIDENCE_INDEX, DETECTION_CONFIDENCE_INDEX + 1)))
            {
                // Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] where Rect2d data style.
                if (_preNMS_box_xywh == null || _preNMS_box_xywh.rows() != num_preNMS)
                {
                    if (_preNMS_box_xywh == null)
                        _preNMS_box_xywh = new Mat();
                    _preNMS_box_xywh.create(num_preNMS, BOUNDING_BOX_COLUMNS, CvType.CV_32FC1);
                }
                using (Mat preNMS_xy1 = preNMS_box.colRange((0, 2)))
                using (Mat preNMS_xy2 = preNMS_box.colRange((2, 4)))
                using (Mat _preNMS_box_xywh_xy = _preNMS_box_xywh.colRange((0, 2)))
                using (Mat _preNMS_box_xywh_wh = _preNMS_box_xywh.colRange((2, 4)))
                {
                    preNMS_xy1.copyTo(_preNMS_box_xywh_xy);
                    Core.subtract(preNMS_xy2, preNMS_xy1, _preNMS_box_xywh_wh);
                }

                if (_NMS_boxes == null || _NMS_boxes.rows() != num_preNMS)
                {
                    if (_NMS_boxes == null)
                        _NMS_boxes = new MatOfRect2d();
                    _NMS_boxes.create(num_preNMS, 1, CvType.CV_64FC4);
                }
                if (_NMS_confidences == null || _NMS_confidences.rows() != num_preNMS)
                {
                    if (_NMS_confidences == null)
                        _NMS_confidences = new MatOfFloat();
                    _NMS_confidences.create(num_preNMS, 1, CvType.CV_32FC1);
                }

                using (Mat boxes_m_c1 = _NMS_boxes.reshape(1, num_preNMS))
                {
                    _preNMS_box_xywh.convertTo(boxes_m_c1, CvType.CV_64F);
                }
                preNMS_confidence.copyTo(_NMS_confidences);


                if (nmsStrategy == NMSStrategy.ClassAgnostic)
                {
                    Dnn.NMSBoxes(_NMS_boxes, _NMS_confidences, score_threshold, nms_threshold, indices, eta, top_k);

                    return;
                }

                if (_NMS_classIds == null || _NMS_classIds.rows() != num_preNMS)
                {
                    if (_NMS_classIds == null)
                        _NMS_classIds = new MatOfInt();
                    _NMS_classIds.create(num_preNMS, 1, CvType.CV_32SC1);
                }

                using (Mat preNMS_classIds = preNMS_Nx6.colRange((DETECTION_CLASS_ID_INDEX, DETECTION_CLASS_ID_INDEX + 1)))
                {
                    preNMS_classIds.convertTo(_NMS_classIds, CvType.CV_32S);
                }

                if (nmsStrategy == NMSStrategy.OpenCVNMSBoxesBatched)
                {
                    Dnn.NMSBoxesBatched(_NMS_boxes, _NMS_confidences, _NMS_classIds, score_threshold, nms_threshold, indices, eta, top_k);
                }
                else if (nmsStrategy == NMSStrategy.ClassWise)
                {
                    DnnProcessingUtils.NMSBoxesClassWise(_NMS_boxes, _NMS_confidences, _NMS_classIds, score_threshold, nms_threshold, indices, eta, top_k);
                }
            }
        }

        protected virtual Mat CreateResultFromBuffer_Nx56(Mat preNMS_Nx56, MatOfInt indices, Mat output0Buffer)
        {
            if (!preNMS_Nx56.isContinuous())
                throw new ArgumentException("preNMS_Nx56 is not continuous.");
            if (!indices.isContinuous())
                throw new ArgumentException("indices is not continuous.");
            if (!output0Buffer.isContinuous())
                throw new ArgumentException("output0Buffer is not continuous.");
            if (indices.rows() > output0Buffer.rows())
                throw new ArgumentException("indices.rows() > output0Buffer.rows()");

            int num = indices.rows();
            Mat result = output0Buffer.rowRange(0, num);

            if (num == 0)
                return result;

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<float> allPreNMS = preNMS_Nx56.AsSpan<float>();
            ReadOnlySpan<int> allIndices = indices.AsSpan<int>();
            Span<float> allResult = result.AsSpan<float>();
#else
            int requiredPreNMSLen = preNMS_Nx56.rows() * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
            int requiredIndicesLen = output0Buffer.rows();
            int requiredResultLen = output0Buffer.rows() * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
            if (_allPreNMSBuffer == null || _allPreNMSBuffer.Length < requiredPreNMSLen)
                _allPreNMSBuffer = new float[requiredPreNMSLen];
            if (_allIndicesBuffer == null || _allIndicesBuffer.Length < requiredIndicesLen)
                _allIndicesBuffer = new int[requiredIndicesLen];
            if (_allResultBuffer == null || _allResultBuffer.Length < requiredResultLen)
                _allResultBuffer = new float[requiredResultLen];

            preNMS_Nx56.get(0, 0, _allPreNMSBuffer);
            indices.get(0, 0, _allIndicesBuffer);
            ReadOnlySpan<float> allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
            ReadOnlySpan<int> allIndices = _allIndicesBuffer.AsSpan(0, requiredIndicesLen);
            Span<float> allResult = _allResultBuffer.AsSpan(0, requiredResultLen);
#endif

            for (int i = 0; i < num; ++i)
            {
                int idx = allIndices[i];
                int resultOffset = i * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
                int preNMSOffset = idx * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);

                // Copy detection and kpts data
                allPreNMS.Slice(preNMSOffset, DETECTION_RESULT_COLUMNS + KPTS_COLUMNS).CopyTo(allResult.Slice(resultOffset, DETECTION_RESULT_COLUMNS + KPTS_COLUMNS));
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            result.put(0, 0, _allResultBuffer);
#endif

            return result;
        }

        protected virtual void ScaleBoxesAndKpts(Mat result, (double width, double height) inputSize, (double width, double height) originalSize)
        {
            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            int num = result.rows();
            if (num == 0)
                return;

            float ratio = Mathf.Max((float)originalSize.width / (float)inputSize.width, (float)originalSize.height / (float)inputSize.height);
            float xFactor = ratio;
            float yFactor = ratio;
            float xShift = ((float)inputSize.width * ratio - (float)originalSize.width) / 2f;
            float yShift = ((float)inputSize.height * ratio - (float)originalSize.height) / 2f;

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<float> allResult = result.AsSpan<float>();
#else
            int requiredResultLen = num * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
            if (_allResultBuffer == null || _allResultBuffer.Length < requiredResultLen)
                _allResultBuffer = new float[requiredResultLen];

            result.get(0, 0, _allResultBuffer);
            Span<float> allResult = _allResultBuffer.AsSpan(0, requiredResultLen);
#endif

            for (int i = 0; i < num; ++i)
            {
                int resultOffset = i * (DETECTION_RESULT_COLUMNS + KPTS_COLUMNS);
                float x1 = Mathf.Round(allResult[resultOffset] * xFactor - xShift);
                float y1 = Mathf.Round(allResult[resultOffset + 1] * yFactor - yShift);
                float x2 = Mathf.Round(allResult[resultOffset + 2] * xFactor - xShift);
                float y2 = Mathf.Round(allResult[resultOffset + 3] * yFactor - yShift);

                allResult[resultOffset] = x1;
                allResult[resultOffset + 1] = y1;
                allResult[resultOffset + 2] = x2;
                allResult[resultOffset + 3] = y2;

                int kptOffset = resultOffset + DETECTION_RESULT_COLUMNS;
                for (int j = 0; j < KPTS_COLUMNS; j += 3)
                {
                    allResult[kptOffset + j] = Mathf.Round(allResult[kptOffset + j] * xFactor - xShift);
                    allResult[kptOffset + j + 1] = Mathf.Round(allResult[kptOffset + j + 1] * yFactor - yShift);
                }
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            result.put(0, 0, _allResultBuffer);
#endif
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
                _poseEstimationNet?.Dispose(); _poseEstimationNet = null;
                _paddedImg?.Dispose(); _paddedImg = null;
                _preNMS_Nx56?.Dispose(); _preNMS_Nx56 = null;
                _preNMS_box_xywh?.Dispose(); _preNMS_box_xywh = null;
                _NMS_boxes?.Dispose(); _NMS_boxes = null;
                _NMS_confidences?.Dispose(); _NMS_confidences = null;
                _NMS_classIds?.Dispose(); _NMS_classIds = null;
                _output0Buffer?.Dispose(); _output0Buffer = null;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
                _allBoxClsconfsKptsBuffer = null;
                _allPreNMSBuffer = null;
                _allIndicesBuffer = null;
                _allResultBuffer = null;
#endif
            }

            base.Dispose(disposing);
        }
    }
}
#endif
