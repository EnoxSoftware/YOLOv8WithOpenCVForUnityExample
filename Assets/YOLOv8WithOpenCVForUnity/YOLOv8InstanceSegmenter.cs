#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
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
    /// YOLOv8 instance segmenter implementation.
    /// This class provides functionality for instance segmentation using the YOLOv8 model implemented with OpenCV's DNN module.
    /// Referring to:
    /// https://github.com/ultralytics/ultralytics/
    ///
    /// [Tested Models]
    /// yolov8n-seg.onnx
    /// yolov9c-seg.onnx
    /// yolov11n-seg.onnx
    /// </summary>
    public class YOLOv8InstanceSegmenter : ProcessingWorkerBase
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
        protected static readonly int BOUNDING_BOX_COLUMNS = 4;
        protected static readonly int DETECTION_RESULT_COLUMNS = 6;
        protected static readonly int DETECTION_BOX_START_INDEX = 0;
        protected static readonly int DETECTION_CONFIDENCE_INDEX = 4;
        protected static readonly int DETECTION_CLASS_ID_INDEX = 5;

        protected Size _inputSize;
        protected float _confThreshold;
        protected float _nmsThreshold;
        protected int _topK;
        protected int _backend;
        protected int _target;
        protected int _numClasses = 80;
        protected int _numMaskCoeff = 32;
        protected int _protoChannels = 32;
        protected int _protoHeight = 160;
        protected int _protoWidth = 160;
        protected Net _segmentPredictionNet;
        protected List<string> _cachedUnconnectedOutLayersNames;
        protected List<string> _classNames;
        protected Mat _paddedImg;
        protected Mat _preNMS_Nx38;
        protected Mat _preNMS_box_xywh;
        protected MatOfRect2d _NMS_boxes;
        protected MatOfFloat _NMS_confidences;
        protected MatOfInt _NMS_classIds;
        protected Mat _output0Buffer;
        protected Mat _output1Buffer;
        protected Mat _maskCoeffsBuffer;

        protected Mat _all_0;
        protected Mat _c_mask;
        protected Mat _maskMat;
        protected Mat _colorMat;
        protected Mat _tempMask;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
        protected float[] _allBoxClsconfsBuffer;
        protected float[] _allMaskCoeffsBuffer;
        protected float[] _allPreNMSBuffer;
        protected int[] _allIndicesBuffer;
        protected float[] _allResultBuffer;
#endif

        /// <summary>
        /// Gets or sets the NMS strategy to use for instance segmentation.
        /// </summary>
        public NMSStrategy SelectedNMSStrategy { get; set; } = NMSStrategy.ClassWise;

        /// <summary>
        /// Initializes a new instance of the YOLOv8InstanceSegmenter class.
        /// </summary>
        /// <param name="modelFilepath">Path to the ONNX model file.</param>
        /// <param name="classesFilepath">Path to the text file containing class names.</param>
        /// <param name="inputSize">Input size for the network (default: 640x640).</param>
        /// <param name="confThreshold">Confidence threshold for filtering detections.</param>
        /// <param name="nmsThreshold">Non-maximum suppression threshold.</param>
        /// <param name="topK">Maximum number of output detections.</param>
        /// <param name="backend">Preferred DNN backend.</param>
        /// <param name="target">Preferred DNN target device.</param>
        public YOLOv8InstanceSegmenter(string modelFilepath, string classesFilepath, Size inputSize,
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
                _segmentPredictionNet = Dnn.readNetFromONNX(modelFilepath);
            }
            catch (Exception e)
            {
                throw new ArgumentException(
                    "Failed to initialize DNN model. Invalid model file path or corrupted model file.", e);
            }
            _segmentPredictionNet.setPreferableBackend(_backend);
            _segmentPredictionNet.setPreferableTarget(_target);
            _cachedUnconnectedOutLayersNames = _segmentPredictionNet.getUnconnectedOutLayersNames();

            _output0Buffer = new Mat(_topK, DETECTION_RESULT_COLUMNS, CvType.CV_32FC1);

            // To get the model information, predict once.
            using (Mat input = new Mat(100, 100, CvType.CV_8UC3))
            {
                Mat inputBlob = PreProcess(input);
                _segmentPredictionNet.setInput(inputBlob);
                List<Mat> outputBlobs = new List<Mat>();
                try
                {
                    _segmentPredictionNet.forward(outputBlobs, _cachedUnconnectedOutLayersNames);
                }
                catch (Exception e)
                {
                    inputBlob.Dispose();
                    throw new ArgumentException(
                        "The input size specified in the constructor may not match the model's expected input size. " +
                        "Please verify the correct input size for your model and update the constructor parameters accordingly.", e);
                }
                _numMaskCoeff = _protoChannels = outputBlobs[1].size(1);
                _protoHeight = outputBlobs[1].size(2);
                _protoWidth = outputBlobs[1].size(3);

                _output1Buffer = new Mat(_topK, _protoHeight * _protoWidth, CvType.CV_32FC1);
                _maskCoeffsBuffer = new Mat(_topK, _numMaskCoeff, CvType.CV_32FC1);
                inputBlob.Dispose();
                for (int i = 0; i < outputBlobs.Count; i++)
                {
                    outputBlobs[i].Dispose();
                }
            }

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
        /// Visualizes detection result on the input image.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The result Mat returned by Detect method.</param>
        /// <param name="printResult">Whether to print result to console.</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public override void Visualize(Mat image, Mat result, bool printResult = false, bool isRGB = false)
        {
            ThrowIfDisposed();

            if (image != null) image.ThrowIfDisposed();
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.cols() < DETECTION_RESULT_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 6 columns.");

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<ObjectDetectionData> data = ToStructuredDataAsSpan(result);
#else
            ObjectDetectionData[] data = ToStructuredData(result);
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
                }

                Debug.Log(sb.ToString());
            }
        }

        /// <summary>
        /// Visualizes segmentation masks on the input image.
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="result">The detection result Mat returned by Detect method.</param>
        /// <param name="masks">The segmentation masks Mat returned by Detect method.</param>
        /// <param name="alpha">Transparency of the visualization (default: 0.5).</param>
        /// <param name="isRGB">Whether the image is in RGB format (vs BGR).</param>
        public virtual void VisualizeMask(Mat image, Mat result, Mat masks, float alpha = 0.5f, bool isRGB = false)
        {
            ThrowIfDisposed();

            if (image != null) image.ThrowIfDisposed();
            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return;
            if (result.cols() < DETECTION_RESULT_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 6 columns.");
            if (masks != null) masks.ThrowIfDisposed();
            if (masks.empty())
                return;

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<ObjectDetectionData> data = ToStructuredDataAsSpan(result);
#else
            ObjectDetectionData[] data = ToStructuredData(result);
#endif

            if (_maskMat == null || _maskMat.width() != image.width() || _maskMat.height() != image.height())
            {
                if (_maskMat == null)
                    _maskMat = new Mat();
                _maskMat.create(image.height(), image.width(), CvType.CV_8UC1);
                _maskMat.setTo(SCALAR_0);
            }
            if (_colorMat == null || _colorMat.width() != image.width() || _colorMat.height() != image.height() || _colorMat.type() != image.type())
            {
                if (_colorMat == null)
                    _colorMat = new Mat();
                _colorMat.create(image.height(), image.width(), image.type());
                _colorMat.setTo(SCALAR_0);
            }

            for (int i = 0; i < data.Length; i++)
            {
                int classId = data[i].ClassId;

                var c = SCALAR_PALETTE[classId % SCALAR_PALETTE.Length].ToValueTuple();
                var color = isRGB ? c : (c.v2, c.v1, c.v0, c.v3);

                using (Mat masks_1xHxW = masks.row(i)) // [1, 160, 160]
                {
                    ScaleAndBinarizeMask(masks_1xHxW, _maskMat, _maskMat.width(), _maskMat.height(), 0.5f);
                    //
                    // _colorMat.setTo(color);
                    // Core.addWeighted(_colorMat, alpha, image, alpha, 0, _colorMat);
                    // _colorMat.copyTo(image, _maskMat);

                    //
                    // or
                    //// use ROI
                    float left = data[i].X1;
                    float top = data[i].Y1;
                    float right = data[i].X2;
                    float bottom = data[i].Y2;
                    OpenCVRect roi_rect = new OpenCVRect((int)left, (int)top, (int)(right - left), (int)(bottom - top));
                    roi_rect = new OpenCVRect(0, 0, image.width(), image.height()).intersect(roi_rect);

                    using (Mat _maskMat_roi = new Mat(_maskMat, roi_rect))
                    using (Mat _colorMat_roi = new Mat(_colorMat, roi_rect))
                    using (Mat _image_roi = new Mat(image, roi_rect))
                    {
                        _colorMat_roi.setTo(color);
                        Core.addWeighted(_colorMat_roi, alpha, _image_roi, alpha, 0, _colorMat_roi);
                        _colorMat_roi.copyTo(_image_roi, _maskMat_roi);
                    }
                    //
                }
            }
        }

        /// <summary>
        /// Detects objects and segmentation masks in the input image.
        /// </summary>
        /// <remarks>
        /// This is a specialized method for object detection and segmentation that:
        /// - Takes a single BGR image as input
        /// - Returns first Mat containing detection result (6 columns per detection)
        /// - Returns second Mat containing segmentation masks (Nx160x160)
        ///
        /// The returned detection result Mat format:
        /// - Each row represents one detection
        /// - Columns: [x1, y1, x2, y2, confidence, classId]
        /// - Use ToStructuredData() or ToStructuredDataAsSpan() to convert to a more convenient format
        ///
        /// The returned segmentation masks Mat format:
        /// - Each row represents one segmentation mask
        /// - Dimensions: [N, 160, 160] where N is the number of detected objects
        /// - Values are in range [0,1] representing the probability of each pixel belonging to the object
        /// - Use ScaleAndBinarizeMask() to convert to a binary mask
        ///
        /// Output options:
        /// - useCopyOutput = false (default): Returns a reference to internal buffer (faster but unsafe across executions)
        /// - useCopyOutput = true: Returns a new copy of the result (thread-safe but slightly slower)
        ///
        /// For better performance in async scenarios, use DetectAsync instead.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="useCopyOutput">Whether to return a copy of the output (true) or a reference (false).</param>
        /// <returns>A tuple containing two Mats: the first is the detection result, and the second is the segmentation masks.</returns>
        public virtual (Mat result, Mat masks) Segment(Mat image, bool useCopyOutput = false)
        {
            Execute(image);
            return useCopyOutput ? (CopyOutput(0), CopyOutput(1)) : (PeekOutput(0), PeekOutput(1));
        }

        /// <summary>
        /// Detects objects and segmentation masks in the input image asynchronously.
        /// </summary>
        /// <remarks>
        /// This is a specialized async method for object detection and segmentation that:
        /// - Takes a single BGR image as input
        /// - Returns first Mat containing detection result (6 columns per detection)
        /// - Returns second Mat containing segmentation masks (Nx160x160)
        ///
        /// The returned detection result Mat format:
        /// - Each row represents one detection
        /// - Columns: [x1, y1, x2, y2, confidence, classId]
        /// - Use ToStructuredData() or ToStructuredDataAsSpan() to convert to a more convenient format
        ///
        /// The returned segmentation masks Mat format:
        /// - Each row represents one segmentation mask
        /// - Dimensions: [N, 160, 160] where N is the number of detected objects
        /// - Values are in range [0,1] representing the probability of each pixel belonging to the object
        /// - Use ScaleAndBinarizeMask() to convert to a binary mask
        ///
        /// Only one detection operation can run at a time.
        /// </remarks>
        /// <param name="image">Input image in BGR format.</param>
        /// <param name="cancellationToken">Optional token to cancel the operation.</param>
        /// <returns>A task that represents the asynchronous operation. The task result contains a tuple of two Mats: the first is the detection result, and the second is the segmentation masks. The caller is responsible for disposing these Mats.</returns>
        public virtual async Task<(Mat result, Mat masks)> SegmentAsync(Mat image, CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(image, cancellationToken);
            return (CopyOutput(0), CopyOutput(1));
        }

        /// <summary>
        /// Converts the detection result matrix to an array of ObjectDetectionData structures.
        /// </summary>
        /// <param name="result">Detection result matrix from Execute method.</param>
        /// <returns>Array of ObjectDetectionData structures containing object detection information.</returns>
        public virtual ObjectDetectionData[] ToStructuredData(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return new ObjectDetectionData[0];
            if (result.cols() < DETECTION_RESULT_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 6 columns.");

            var dst = new ObjectDetectionData[result.rows()];
            MatUtils.copyFromMat(result, dst);

            return dst;
        }

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
        /// <summary>
        /// Converts the detection result matrix to a span of ObjectDetectionData structures.
        /// </summary>
        /// <param name="result">Detection result matrix from Execute method.</param>
        /// <returns>Span of ObjectDetectionData structures containing object detection information.</returns>
        public virtual Span<ObjectDetectionData> ToStructuredDataAsSpan(Mat result)
        {
            ThrowIfDisposed();

            if (result != null) result.ThrowIfDisposed();
            if (result.empty())
                return Span<ObjectDetectionData>.Empty;
            if (result.cols() < DETECTION_RESULT_COLUMNS)
                throw new ArgumentException("Invalid result matrix. It must have at least 6 columns.");
            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");

            return result.AsSpan<ObjectDetectionData>();
        }
#endif

        /// <summary>
        /// Scales and binarizes the segmentation mask to the specified size.
        /// </summary>
        /// <param name="mask">Segmentation mask in float format with shape [1,160,160]</param>
        /// <param name="dst">Output mask in 8-bit format</param>
        /// <param name="width">Target width</param>
        /// <param name="height">Target height</param>
        /// <param name="threshold">Binarization threshold (default: 0.5)</param>
        public virtual void ScaleAndBinarizeMask(Mat mask, Mat dst, int width, int height, float threshold = 0.5f)
        {
            if (mask != null) mask.ThrowIfDisposed();
            if (mask.size(0) != 1 || mask.size(1) != _protoHeight || mask.size(2) != _protoWidth)
                throw new ArgumentException("Invalid mask matrix.");
            if (dst != null) dst.ThrowIfDisposed();

            // Reshape mask to [160,160]
            using (Mat mask_160x160 = mask.reshape(1, _protoHeight))
            {
                // Calculate scaling and padding considering the input size
                float scale = Mathf.Min((float)_protoWidth / width, (float)_protoHeight / height);
                int mask_w = (int)(width * scale);
                int mask_h = (int)(height * scale);
                int mask_pad_x = (_protoWidth - mask_w) / 2;
                int mask_pad_y = (_protoHeight - mask_h) / 2;

                if (_tempMask == null || _tempMask.width() != width || _tempMask.height() != height)
                {
                    if (_tempMask == null)
                        _tempMask = new Mat();
                    _tempMask.create(height, width, CvType.CV_32FC1);
                    _tempMask.setTo(SCALAR_0);
                }

                // Create temporary mask
                using (Mat roi = new Mat(mask_160x160, (mask_pad_x, mask_pad_y, mask_w, mask_h)))
                {
                    // Resize mask
                    Imgproc.resize(roi, _tempMask, (width, height), 0, 0, Imgproc.INTER_LINEAR);

                    // Binarize
                    Imgproc.threshold(_tempMask, _tempMask, threshold, 1.0, Imgproc.THRESH_BINARY);

                    // Convert to 8-bit format
                    _tempMask.convertTo(dst, CvType.CV_8U, 255.0);
                }
            }
        }

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
            _segmentPredictionNet.setInput(inputBlob);
            List<Mat> outputBlobs = new List<Mat>();
            try
            {
                _segmentPredictionNet.forward(outputBlobs, _cachedUnconnectedOutLayersNames);
            }
            catch (Exception e)
            {
                inputBlob.Dispose();
                throw new ArgumentException(
                    "The input size specified in the constructor may not match the model's expected input size. " +
                    "Please verify the correct input size for your model and update the constructor parameters accordingly.", e);
            }

            // Postprocess
            (Mat result, Mat masks) = PostProcess(outputBlobs, image.sizeAsValueTuple()); // submat of _output0Buffer and _output1Buffer is returned

            // Any rewriting of buffer data must be done within the lock statement
            // Do not return the buffer itself because it will be destroyed,
            // but return a submat of the same size as the result extracted using rowRange

            inputBlob.Dispose();
            for (int i = 0; i < outputBlobs.Count; i++)
            {
                outputBlobs[i].Dispose();
            }

            // [n, 6] (xyxy, conf, cls)
            // [n, 160, 160] (masks)
            return new Mat[] { result, masks };
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

        protected virtual (Mat result, Mat masks) PostProcess(List<Mat> outputBlobs, (double width, double height) originalSize)
        {
            Mat outputBlob_0 = outputBlobs[0]; // 1 * (4 + _numClasses + _numMaskCoeff) * N
            Mat outputBlob_1 = outputBlobs[1]; // 1 * _protoChannels * _protoHeight * _protoWidth // prototype masks

            if (outputBlob_0.size(1) != 4 + _numClasses + _numMaskCoeff)
            {
                Debug.LogWarning("The number of classes and output shapes are different. " +
                                "( outputBlob_0.size(1):" + outputBlob_0.size(1) + " != 4 + numClasses:" + _numClasses + " + _numMaskCoeff:" + _numMaskCoeff + " )\n" +
                                "When using a custom model, be sure to set the correct number of classes by loading the appropriate custom classesFile.");

                _numClasses = outputBlob_0.size(1) - 4 - _numMaskCoeff;
            }

            int num = outputBlob_0.size(1);
            using (Mat outputBlob_NxBoxConfClsconfsMaskcoeff = outputBlob_0.reshape(1, num))
            {
                // pre-NMS
                PreNMS_BoxClsconfsMaskcoeff_CXCYWHtoX1Y1X2Y2_YOLOv8(outputBlob_NxBoxConfClsconfsMaskcoeff, ref _preNMS_Nx38, _confThreshold, _topK);
            }

            Mat result = null;
            Mat masks = null;
            using (MatOfInt indices = new MatOfInt())
            {
                // NMS
                NMS(_preNMS_Nx38, _confThreshold, _nmsThreshold, indices, 1f, _topK, SelectedNMSStrategy);

                lock (_lockObject)
                {
                    // Create result
                    result = CreateResultFromBuffer_Nx38(_preNMS_Nx38, indices, _output0Buffer);

                    // Create masks
                    masks = CreateMasksFromBuffer(outputBlob_1, _preNMS_Nx38, indices, _output1Buffer);
                    CropMasks(result, masks, _inputSize.ToValueTuple());

                    // Scale boxes
                    ScaleBoxes(result, _inputSize.ToValueTuple(), originalSize);
                }
            }

            // [
            //   [xyxy, conf, cls]
            //   ...
            //   [xyxy, conf, cls]
            // ]

            // [
            //   [mask]
            //   ...
            //   [mask]
            // ]
            return (result, masks);
        }

        protected virtual void PreNMS_BoxClsconfsMaskcoeff_CXCYWHtoX1Y1X2Y2_YOLOv8(Mat boxClsconfsMaskcoeff, ref Mat preNMS_Nx38, float score_threshold, int top_k = 300)
        {
            if (!boxClsconfsMaskcoeff.isContinuous())
                throw new ArgumentException("boxClsconfsMaskcoeff is not continuous.");

            const int CLASS_CONFIDENCES_INDEX = 4;

            int boxClsconfsMaskcoeff_rows = boxClsconfsMaskcoeff.rows();
            int boxClsconfsMaskcoeff_cols = boxClsconfsMaskcoeff.cols();

            if (preNMS_Nx38 == null || preNMS_Nx38.rows() < top_k)
            {
                if (preNMS_Nx38 == null)
                    preNMS_Nx38 = new Mat();
                preNMS_Nx38.create(top_k, DETECTION_RESULT_COLUMNS + _numMaskCoeff, CvType.CV_32FC1);
                preNMS_Nx38.setTo(SCALAR_0);
            }
            int num_preNMS = preNMS_Nx38.rows();

            // Initialize output data (confidences only)
            using (Mat preNMS_Nx38_col_4 = preNMS_Nx38.col(DETECTION_CONFIDENCE_INDEX))
            {
                preNMS_Nx38_col_4.setTo(SCALAR_0);
            }

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<float> allBoxClsconfsMaskcoeff = boxClsconfsMaskcoeff.AsSpan<float>();
            Span<float> allPreNMS = preNMS_Nx38.AsSpan<float>();
#else
            int requiredBoxClsconfsMaskcoeffLen = boxClsconfsMaskcoeff_rows * boxClsconfsMaskcoeff_cols;
            int requiredPreNMSLen = num_preNMS * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);

            if (_allBoxClsconfsBuffer == null || _allBoxClsconfsBuffer.Length < requiredBoxClsconfsMaskcoeffLen)
                _allBoxClsconfsBuffer = new float[requiredBoxClsconfsMaskcoeffLen];
            if (_allPreNMSBuffer == null || _allPreNMSBuffer.Length < requiredPreNMSLen)
                _allPreNMSBuffer = new float[requiredPreNMSLen];

            boxClsconfsMaskcoeff.get(0, 0, _allBoxClsconfsBuffer);
            preNMS_Nx38.get(0, 0, _allPreNMSBuffer);
            ReadOnlySpan<float> allBoxClsconfsMaskcoeff = _allBoxClsconfsBuffer.AsSpan(0, requiredBoxClsconfsMaskcoeffLen);
            Span<float> allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
#endif

            int ind = 0;
            for (int i = 0; i < boxClsconfsMaskcoeff_cols; ++i)
            {
                float maxVal = float.MinValue;
                int maxIdx = -1;
                for (int j = 0; j < _numClasses; ++j)
                {
                    float conf = allBoxClsconfsMaskcoeff[(CLASS_CONFIDENCES_INDEX + j) * boxClsconfsMaskcoeff_cols + i];
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
                        Mat new_preNMS_Nx38 = new Mat(num_preNMS * 2, DETECTION_RESULT_COLUMNS + _numMaskCoeff, CvType.CV_32FC1, SCALAR_0.ToValueTuple());
                        using (Mat new_preNMS_Nx38_roi = new_preNMS_Nx38.rowRange(0, num_preNMS))
                        {
                            preNMS_Nx38.copyTo(new_preNMS_Nx38_roi);
                        }
                        preNMS_Nx38.Dispose();
                        preNMS_Nx38 = new_preNMS_Nx38;
                        num_preNMS = preNMS_Nx38.rows();

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                        allPreNMS = preNMS_Nx38.AsSpan<float>();
#else
                        requiredPreNMSLen = num_preNMS * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);
                        float[] newBuffer = new float[requiredPreNMSLen];
                        Array.Copy(_allPreNMSBuffer, newBuffer, _allPreNMSBuffer.Length);
                        _allPreNMSBuffer = newBuffer;
                        allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
#endif
                    }

                    int preNMSIdx = ind * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);

                    // Convert from [cx,cy,w,h] to [x1,y1,x2,y2]
                    float cx = allBoxClsconfsMaskcoeff[0 * boxClsconfsMaskcoeff_cols + i];
                    float cy = allBoxClsconfsMaskcoeff[1 * boxClsconfsMaskcoeff_cols + i];
                    float w = allBoxClsconfsMaskcoeff[2 * boxClsconfsMaskcoeff_cols + i];
                    float h = allBoxClsconfsMaskcoeff[3 * boxClsconfsMaskcoeff_cols + i];

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

                    // Copy mask coefficients data
                    int maskCoeffIdx = (CLASS_CONFIDENCES_INDEX + _numClasses) * boxClsconfsMaskcoeff_cols + i;
                    for (int k = 0; k < _numMaskCoeff; ++k)
                    {
                        allPreNMS[preNMSIdx + DETECTION_RESULT_COLUMNS + k] = allBoxClsconfsMaskcoeff[maskCoeffIdx + k * boxClsconfsMaskcoeff_cols];
                    }

                    ind++;
                }
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            preNMS_Nx38.put(0, 0, _allPreNMSBuffer);
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

        protected virtual Mat CreateResultFromBuffer_Nx38(Mat preNMS_Nx38, MatOfInt indices, Mat output0Buffer)
        {
            if (!preNMS_Nx38.isContinuous())
                throw new ArgumentException("preNMS_Nx38 is not continuous.");
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
            ReadOnlySpan<float> allPreNMS = preNMS_Nx38.AsSpan<float>();
            ReadOnlySpan<int> allIndices = indices.AsSpan<int>();
            Span<float> allResult = result.AsSpan<float>();
#else
            int requiredPreNMSLen = preNMS_Nx38.rows() * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);
            int requiredIndicesLen = output0Buffer.rows();
            int requiredResultLen = output0Buffer.rows() * DETECTION_RESULT_COLUMNS;
            if (_allPreNMSBuffer == null || _allPreNMSBuffer.Length < requiredPreNMSLen)
                _allPreNMSBuffer = new float[requiredPreNMSLen];
            if (_allIndicesBuffer == null || _allIndicesBuffer.Length < requiredIndicesLen)
                _allIndicesBuffer = new int[requiredIndicesLen];
            if (_allResultBuffer == null || _allResultBuffer.Length < requiredResultLen)
                _allResultBuffer = new float[requiredResultLen];

            preNMS_Nx38.get(0, 0, _allPreNMSBuffer);
            indices.get(0, 0, _allIndicesBuffer);
            ReadOnlySpan<float> allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
            ReadOnlySpan<int> allIndices = _allIndicesBuffer.AsSpan(0, requiredIndicesLen);
            Span<float> allResult = _allResultBuffer.AsSpan(0, requiredResultLen);
#endif

            for (int i = 0; i < num; ++i)
            {
                int idx = allIndices[i];
                int resultOffset = i * DETECTION_RESULT_COLUMNS;
                int preNMSOffset = idx * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);

                // Copy detection data
                allPreNMS.Slice(preNMSOffset, DETECTION_RESULT_COLUMNS).CopyTo(allResult.Slice(resultOffset, DETECTION_RESULT_COLUMNS));
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            result.put(0, 0, _allResultBuffer);
#endif

            return result;
        }

        protected virtual void ScaleBoxes(Mat result, (double width, double height) inputSize, (double width, double height) originalSize)
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
            int requiredResultLen = num * DETECTION_RESULT_COLUMNS;
            if (_allResultBuffer == null || _allResultBuffer.Length < requiredResultLen)
                _allResultBuffer = new float[requiredResultLen];

            result.get(0, 0, _allResultBuffer);
            Span<float> allResult = _allResultBuffer.AsSpan(0, requiredResultLen);
#endif

            for (int i = 0; i < num; ++i)
            {
                int resultOffset = i * DETECTION_RESULT_COLUMNS;
                float x1 = Mathf.Round(allResult[resultOffset] * xFactor - xShift);
                float y1 = Mathf.Round(allResult[resultOffset + 1] * yFactor - yShift);
                float x2 = Mathf.Round(allResult[resultOffset + 2] * xFactor - xShift);
                float y2 = Mathf.Round(allResult[resultOffset + 3] * yFactor - yShift);

                allResult[resultOffset] = x1;
                allResult[resultOffset + 1] = y1;
                allResult[resultOffset + 2] = x2;
                allResult[resultOffset + 3] = y2;
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            result.put(0, 0, _allResultBuffer);
#endif
        }

        protected virtual Mat CreateMasksFromBuffer(Mat protoMasks, Mat preNMS_Nx38, MatOfInt indices, Mat output1Buffer)
        {
            if (!preNMS_Nx38.isContinuous())
                throw new ArgumentException("preNMS_Nx38 is not continuous.");
            if (!indices.isContinuous())
                throw new ArgumentException("indices is not continuous.");
            if (!output1Buffer.isContinuous())
                throw new ArgumentException("output1Buffer is not continuous.");
            if (indices.rows() > output1Buffer.rows())
                throw new ArgumentException("indices.rows() > output1Buffer.rows()");

            int num = indices.rows();
            Mat masks = output1Buffer.rowRange(0, num);

            if (num == 0)
                return masks;

            Mat maskCoeffs = _maskCoeffsBuffer.rowRange(0, num);

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<float> allPreNMS = preNMS_Nx38.AsSpan<float>();
            ReadOnlySpan<int> allIndices = indices.AsSpan<int>();
            Span<float> allMaskCoeffs = maskCoeffs.AsSpan<float>();
#else
            int requiredPreNMSLen = preNMS_Nx38.rows() * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);
            int requiredIndicesLen = output1Buffer.rows();
            int requiredMaskCoeffsLen = output1Buffer.rows() * _numMaskCoeff;
            if (_allPreNMSBuffer == null || _allPreNMSBuffer.Length < requiredPreNMSLen)
                _allPreNMSBuffer = new float[requiredPreNMSLen];
            if (_allIndicesBuffer == null || _allIndicesBuffer.Length < requiredIndicesLen)
                _allIndicesBuffer = new int[requiredIndicesLen];
            if (_allMaskCoeffsBuffer == null || _allMaskCoeffsBuffer.Length < requiredMaskCoeffsLen)
                _allMaskCoeffsBuffer = new float[requiredMaskCoeffsLen];

            preNMS_Nx38.get(0, 0, _allPreNMSBuffer);
            indices.get(0, 0, _allIndicesBuffer);
            ReadOnlySpan<float> allPreNMS = _allPreNMSBuffer.AsSpan(0, requiredPreNMSLen);
            ReadOnlySpan<int> allIndices = _allIndicesBuffer.AsSpan(0, requiredIndicesLen);
            Span<float> allMaskCoeffs = _allMaskCoeffsBuffer.AsSpan(0, requiredMaskCoeffsLen);
#endif

            for (int i = 0; i < num; ++i)
            {
                int idx = allIndices[i];
                int maskCoeffOffset = i * _numMaskCoeff;
                int preNMSOffset = idx * (DETECTION_RESULT_COLUMNS + _numMaskCoeff);

                // Copy mask coefficients data
                allPreNMS.Slice(preNMSOffset + DETECTION_RESULT_COLUMNS, _numMaskCoeff).CopyTo(allMaskCoeffs.Slice(maskCoeffOffset, _numMaskCoeff));
            }

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
            maskCoeffs.put(0, 0, _allMaskCoeffsBuffer);
#endif

            using (Mat protoMasks_32x25600 = protoMasks.reshape(1, _numMaskCoeff)) // [1, 32, 160, 160] => [32, 25600]
            {
                Core.gemm(maskCoeffs, protoMasks_32x25600, 1, new Mat(), 0, masks);
            }

            Sigmoid(masks);

            Mat masks_reshaped = masks.reshape(1, new int[] { num, _protoHeight, _protoWidth }); // [num, 160, 160]
            masks.Dispose();

            return masks_reshaped;
        }

        protected virtual void CropMasks(Mat result, Mat masks, (double width, double height) inputSize)
        {
            if (!result.isContinuous())
                throw new ArgumentException("result is not continuous.");
            if (!masks.isContinuous())
                throw new ArgumentException("masks is not continuous.");
            if (result.rows() != masks.size(0))
                throw new ArgumentException("result.rows() != masks.rows()");

            int num = result.rows();
            if (num == 0)
                return;

            float ratio = Mathf.Max(_protoWidth / (float)inputSize.width, _protoHeight / (float)inputSize.height);

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<float> allResult = result.AsSpan<float>();
#else
            int requiredResultLen = num * DETECTION_RESULT_COLUMNS;
            if (_allResultBuffer == null || _allResultBuffer.Length < requiredResultLen)
                _allResultBuffer = new float[requiredResultLen];

            result.get(0, 0, _allResultBuffer);
            ReadOnlySpan<float> allResult = _allResultBuffer.AsSpan(0, requiredResultLen);
#endif

            if (_all_0 == null)
                _all_0 = new Mat(_protoHeight, _protoWidth, CvType.CV_32FC1, SCALAR_0); // [160, 160]

            if (_c_mask == null)
                _c_mask = new Mat(_protoHeight, _protoWidth, CvType.CV_8UC1); // [160, 160]

            for (int i = 0; i < num; i++)
            {
                int resultOffset = i * DETECTION_RESULT_COLUMNS;
                float x1 = allResult[resultOffset];
                float y1 = allResult[resultOffset + 1];
                float x2 = allResult[resultOffset + 2];
                float y2 = allResult[resultOffset + 3];

                float x1_crop = x1 * ratio;
                float y1_crop = y1 * ratio;
                float x2_crop = x2 * ratio;
                float y2_crop = y2 * ratio;

                using (Mat masks_row_i = masks.row(i))
                using (Mat masks_row_i_reshape = masks_row_i.reshape(1, _protoHeight)) // [160, 160]
                {
                    _c_mask.setTo(SCALAR_1);
                    OpenCVRect roi_rect = new OpenCVRect((int)x1_crop, (int)y1_crop, (int)(x2_crop - x1_crop), (int)(y2_crop - y1_crop));
                    OpenCVRect bounds = new OpenCVRect(0, 0, _c_mask.width(), _c_mask.height());
                    roi_rect = roi_rect.intersect(bounds);
                    using (Mat roi = new Mat(_c_mask, roi_rect))
                    {
                        roi.setTo(SCALAR_0);
                    }
                    _all_0.copyTo(masks_row_i_reshape, _c_mask);
                }
            }
        }

        protected virtual void Sigmoid(Mat mat)
        {
            if (mat == null)
                throw new ArgumentNullException("mat");
            if (mat != null)
                mat.ThrowIfDisposed();

            //python: 1 / (1 + np.exp(-x))

            Core.multiply(mat, (-1, -1, -1, -1), mat);  // -x
            Core.exp(mat, mat);                         // exp(-x)
            Core.add(mat, (1, 1, 1, 1), mat);           // 1 + exp(-x)
            Core.divide(1.0, mat, mat);                 // 1 / (1 + exp(-x))
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
                _segmentPredictionNet?.Dispose(); _segmentPredictionNet = null;
                _paddedImg?.Dispose(); _paddedImg = null;
                _preNMS_Nx38?.Dispose(); _preNMS_Nx38 = null;
                _preNMS_box_xywh?.Dispose(); _preNMS_box_xywh = null;
                _NMS_boxes?.Dispose(); _NMS_boxes = null;
                _NMS_confidences?.Dispose(); _NMS_confidences = null;
                _NMS_classIds?.Dispose(); _NMS_classIds = null;
                _output0Buffer?.Dispose(); _output0Buffer = null;
                _output1Buffer?.Dispose(); _output1Buffer = null;
                _maskCoeffsBuffer?.Dispose(); _maskCoeffsBuffer = null;

                _all_0?.Dispose(); _all_0 = null;
                _c_mask?.Dispose(); _c_mask = null;
                _maskMat?.Dispose(); _maskMat = null;
                _colorMat?.Dispose(); _colorMat = null;
                _tempMask?.Dispose(); _tempMask = null;

#if !NET_STANDARD_2_1 || OPENCV_DONT_USE_UNSAFE_CODE
                _allBoxClsconfsBuffer = null;
                _allMaskCoeffsBuffer = null;
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
