#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using YOLOv8WithOpenCVForUnity.Worker;

namespace YOLOv8WithOpenCVForUnityExample
{
    /// <summary>
    /// YOLOv8 Pose Estimation Example
    /// Referring to:
    /// https://github.com/ultralytics/ultralytics/
    ///
    /// [Tested Models]
    /// yolov8n-pose.onnx
    /// yolov11n-pose.onnx
    /// </summary>
    [RequireComponent(typeof(MultiSource2MatHelper))]
    public class YOLOv8PoseEstimationExample : MonoBehaviour
    {
        [Header("Output")]
        [Tooltip("The RawImage for previewing the result.")]
        public RawImage _resultPreview;

        [Header("UI")]
        public Toggle _useAsyncInferenceToggle;
        public bool _useAsyncInference = false;

        [Header("Model Settings")]
        [Tooltip("Path to a binary file of model contains trained weights.")]
        public string _model = "YOLOv8WithOpenCVForUnityExample/yolov8n-pose.onnx";

        [Tooltip("Optional path to a text file with names of classes to label detected objects.")]
        public string _classes = "";

        [Tooltip("Confidence threshold.")]
        public float _confThreshold = 0.25f;

        [Tooltip("Non-maximum suppression threshold.")]
        public float _nmsThreshold = 0.45f;

        [Tooltip("Maximum detections per image.")]
        public int _topK = 300;

        [Tooltip("Preprocess input image by resizing to a specific width.")]
        public int _inpWidth = 640;

        [Tooltip("Preprocess input image by resizing to a specific height.")]
        public int _inpHeight = 640;

        private YOLOv8PoseEstimater _poseEstimator;
        private string _classes_filepath;
        private string _model_filepath;

        private Texture2D _texture;
        private MultiSource2MatHelper _multiSource2MatHelper;
        private Mat _bgrMat;

        private FpsMonitor _fpsMonitor;
        private CancellationTokenSource _cts = new CancellationTokenSource();

        private Mat _bgrMatForAsync;
        private Mat _latestDetectedObjects;
        private Task _inferenceTask;
        private readonly Queue<Action> _mainThreadQueue = new();
        private readonly object _queueLock = new();

        // Use this for initialization
        async void Start()
        {
            _fpsMonitor = GetComponent<FpsMonitor>();

            _multiSource2MatHelper = gameObject.GetComponent<MultiSource2MatHelper>();
            _multiSource2MatHelper.OutputColorFormat = Source2MatHelperColorFormat.RGBA;

            // Update GUI state
#if !UNITY_WEBGL || UNITY_EDITOR
            _useAsyncInferenceToggle.isOn = _useAsyncInference;
#else
            _useAsyncInferenceToggle.isOn = false;
            _useAsyncInferenceToggle.interactable = false;
#endif

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            if (_fpsMonitor != null)
                _fpsMonitor.ConsoleText = "Preparing file access...";

            if (!string.IsNullOrEmpty(_classes))
            {
                _classes_filepath = await OpenCVEnv.GetFilePathTaskAsync(_classes, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_classes_filepath)) Debug.Log("The file:" + _classes + " did not exist.");
            }
            if (!string.IsNullOrEmpty(_model))
            {
                _model_filepath = await OpenCVEnv.GetFilePathTaskAsync(_model, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_model_filepath)) Debug.Log("The file:" + _model + " did not exist.");
            }

            if (_fpsMonitor != null)
                _fpsMonitor.ConsoleText = "";

            Run();
        }

        // Use this for initialization
        protected virtual void Run()
        {
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            OpenCVDebug.SetDebugMode(true);


            if (string.IsNullOrEmpty(_model_filepath))
            {
                Debug.LogError("model: " + _model + " or " + "classes: " + _classes + " is not loaded.");
            }
            else
            {
                _poseEstimator = new YOLOv8PoseEstimater(_model_filepath, _classes_filepath, new Size(_inpWidth, _inpHeight), _confThreshold, _nmsThreshold, _topK);
            }

            _multiSource2MatHelper.Initialize();
        }

        /// <summary>
        /// Raises the source to mat helper initialized event.
        /// </summary>
        public virtual void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = _multiSource2MatHelper.GetMat();

            _texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);
            OpenCVMatUtils.MatToTexture2D(rgbaMat, _texture);

            _resultPreview.texture = _texture;
            _resultPreview.GetComponent<AspectRatioFitter>().aspectRatio = (float)_texture.width / _texture.height;


            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("width", rgbaMat.width().ToString());
                _fpsMonitor.Add("height", rgbaMat.height().ToString());
                _fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }

            _bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
            _bgrMatForAsync = new Mat();
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public virtual void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            if (_inferenceTask != null && !_inferenceTask.IsCompleted) _inferenceTask.Wait(500);

            _bgrMat?.Dispose(); _bgrMat = null;

            _bgrMatForAsync?.Dispose(); _bgrMatForAsync = null;
            _latestDetectedObjects?.Dispose(); _latestDetectedObjects = null;

            if (_texture != null) Texture2D.Destroy(_texture); _texture = null;
        }

        /// <summary>
        /// Raises the source to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        /// <param name="message">Message.</param>
        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode errorCode, string message)
        {
            Debug.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);

            if (_fpsMonitor != null)
            {
                _fpsMonitor.ConsoleText = "ErrorCode: " + errorCode + ":" + message;
            }
        }

        // Update is called once per frame
        void Update()
        {
            ProcessMainThreadQueue();

            if (_multiSource2MatHelper.IsPlaying() && _multiSource2MatHelper.DidUpdateThisFrame())
            {

                Mat rgbaMat = _multiSource2MatHelper.GetMat();

                if (_poseEstimator == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {
                    Imgproc.cvtColor(rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);

                    if (_useAsyncInference)
                    {
                        // asynchronous execution

                        if (_inferenceTask == null || _inferenceTask.IsCompleted)
                        {
                            _bgrMat.copyTo(_bgrMatForAsync); // for asynchronous execution, deep copy
                            _inferenceTask = Task.Run(async () =>
                            {
                                try
                                {
                                    // Pose estimator inference
                                    var newObjects = await _poseEstimator.EstimateAsync(_bgrMatForAsync);
                                    RunOnMainThread(() =>
                                        {
                                            _latestDetectedObjects?.Dispose();
                                            _latestDetectedObjects = newObjects;
                                        });
                                }
                                catch (OperationCanceledException ex)
                                {
                                    Debug.Log($"Inference canceled: {ex}");
                                }
                                catch (Exception ex)
                                {
                                    Debug.LogError($"Inference error: {ex}");
                                }
                            });
                        }

                        Imgproc.cvtColor(_bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                        if (_latestDetectedObjects != null)
                        {
                            _poseEstimator.Visualize(rgbaMat, _latestDetectedObjects, false, true);
                        }
                    }
                    else
                    {
                        // synchronous execution

                        // TickMeter tm = new TickMeter();
                        // tm.start();

                        // Pose estimator inference
                        using (Mat objects = _poseEstimator.Estimate(_bgrMat))
                        {
                            // tm.stop();
                            // Debug.Log("YOLOv8PoseEstimater Inference time, ms: " + tm.getTimeMilli());

                            Imgproc.cvtColor(_bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                            _poseEstimator.Visualize(rgbaMat, objects, false, true);
                        }
                    }
                }

                OpenCVMatUtils.MatToTexture2D(rgbaMat, _texture);
            }
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        protected virtual void OnDestroy()
        {
            _multiSource2MatHelper?.Dispose();

            _poseEstimator?.Dispose();

            OpenCVDebug.SetDebugMode(false);

            _cts?.Dispose();
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public virtual void OnBackButtonClick()
        {
            SceneManager.LoadScene("YOLOv8WithOpenCVForUnityExample");
        }

        /// <summary>
        /// Raises the play button click event.
        /// </summary>
        public virtual void OnPlayButtonClick()
        {
            _multiSource2MatHelper.Play();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public virtual void OnPauseButtonClick()
        {
            _multiSource2MatHelper.Pause();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public virtual void OnStopButtonClick()
        {
            _multiSource2MatHelper.Stop();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public virtual void OnChangeCameraButtonClick()
        {
            _multiSource2MatHelper.RequestedIsFrontFacing = !_multiSource2MatHelper.RequestedIsFrontFacing;
        }

        /// <summary>
        /// Raises the use async inference toggle value changed event.
        /// </summary>
        public void OnUseAsyncInferenceToggleValueChanged()
        {
            if (_useAsyncInferenceToggle.isOn != _useAsyncInference)
            {
                // Wait for inference to complete before changing the toggle
                if (_inferenceTask != null && !_inferenceTask.IsCompleted) _inferenceTask.Wait(500);

                _useAsyncInference = _useAsyncInferenceToggle.isOn;
            }
        }

        private void RunOnMainThread(Action action)
        {
            if (action == null) return;

            lock (_queueLock)
            {
                _mainThreadQueue.Enqueue(action);
            }
        }

        private void ProcessMainThreadQueue()
        {
            while (true)
            {
                Action action = null;
                lock (_queueLock)
                {
                    if (_mainThreadQueue.Count == 0)
                        break;

                    action = _mainThreadQueue.Dequeue();
                }

                try { action?.Invoke(); }
                catch (Exception ex) { Debug.LogException(ex); }
            }
        }
    }
}
#endif
