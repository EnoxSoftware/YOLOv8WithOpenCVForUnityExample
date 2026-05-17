#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using OpenCVForUnity.UnityIntegration.Runner;
using OpenCVForUnity.UnityIntegration.Worker;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using OpenCVForUnity.UnityIntegration.Worker.Utils;
#if OPENCV_SENTIS_AVAILABLE
using Unity.InferenceEngine;
#endif
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using YOLOv8WithOpenCVForUnity.Worker;
using static OpenCVForUnity.UnityIntegration.Helper.Source2Mat.MultiSource2MatHelper;

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
        // Public Fields
        [Header("Output")]
        [Tooltip("The RawImage for previewing the result.")]
        public RawImage ResultPreview;

        [Header("UI")]
        [Tooltip("ON: Sentis. OFF: OpenCV DNN. Assign OnUseSentisInferenceToggleValueChanged to this toggle's On Value Changed in the Inspector.")]
        public Toggle UseSentisInferenceToggle;
        [Tooltip("Sentis backend selector. Dropdown option order must match Enum.GetValues(typeof(BackendType)) (numeric order). Assign OnSentisBackendDropdownValueChanged to On Value Changed (int). Value changes reinitialize inference.")]
        public Dropdown SentisBackendDropdown;
#if OPENCV_SENTIS_AVAILABLE
        [Tooltip("When enabled, runs YOLOv8 inference with Sentis (MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS). Inspector paths may stay .onnx; at runtime they are rewritten to .sentis and loaded from StreamingAssets (place a matching .sentis beside the onnx file).")]
        public bool UseSentisInference = true;
        [Tooltip("When using Sentis: backend / target selects Sentis BackendType (CPU / GPU, etc.).")]
        public BackendType YoloSentisBackendType = BackendType.GPUCompute;
#endif
        public Toggle UseAsyncInferenceToggle;
        public bool UseAsyncInference = true;

        [Header("Model Settings")]
        [Tooltip("Path to a binary file of model contains trained weights.")]
        public string Model = "YOLOv8WithOpenCVForUnityExample/yolov8n-pose.onnx";

        [Tooltip("Optional path to a text file with names of classes to label detected objects.")]
        public string Classes = "";

        [Tooltip("Confidence threshold.")]
        public float ConfThreshold = 0.25f;

        [Tooltip("Non-maximum suppression threshold.")]
        public float NmsThreshold = 0.45f;

        [Tooltip("Maximum detections per image.")]
        public int TopK = 300;

        [Tooltip("Preprocess input image by resizing to a specific width.")]
        public int InpWidth = 640;

        [Tooltip("Preprocess input image by resizing to a specific height.")]
        public int InpHeight = 640;

        // Private Fields
        private YOLOv8PoseEstimater _poseEstimator;
        private string _classesFilepath;
        private string _modelFilepathOnnx;
#if OPENCV_SENTIS_AVAILABLE
        private string _modelFilepathSentis;
        /// <summary>
        /// <see cref="BackendType"/> values in <see cref="Enum.GetValues(System.Type)"/> order (sorted by underlying numeric value). Dropdown options must use the same order.
        /// </summary>
        private static readonly BackendType[] SentisBackendTypesInEnumOrder =
            (BackendType[])Enum.GetValues(typeof(BackendType));
#endif

        private Texture2D _texture;
        private MultiSource2MatHelper _multiSource2MatHelper;
        private Mat _bgrMat;

        private FpsMonitor _fpsMonitor;
        private CancellationTokenSource _cts = new CancellationTokenSource();
        private MatSingleFlightSyncAsyncRunner _inferenceRunner;
        private bool _inferenceReinitializing;

        private async void Start()
        {
            _fpsMonitor = GetComponent<FpsMonitor>();

            _multiSource2MatHelper = gameObject.GetComponent<MultiSource2MatHelper>();

#if UNITY_6000_0_OR_NEWER
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.WebGPU && _multiSource2MatHelper.RequestedSource2MatHelperClassName == MultiSource2MatHelperClassName.WebCamTexture2MatHelper)
            {
                _multiSource2MatHelper.RequestedSource2MatHelperClassName = MultiSource2MatHelperClassName.WebCamTexture2MatAsyncGPUHelper;
            }
#endif
            _multiSource2MatHelper.OutputColorFormat = Source2MatHelperColorFormat.RGBA;

            UpdateUseSentisInference();
            UpdateUseAsyncInference();
            UpdateInferenceModeToggles(inferenceReinitializing: false);

            if (_fpsMonitor != null)
                _fpsMonitor.ConsoleText = "Preparing file access...";

            if (!string.IsNullOrEmpty(Classes))
            {
                _classesFilepath = await OpenCVEnv.GetFilePathTaskAsync(Classes, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_classesFilepath)) Debug.Log("The file:" + Classes + " did not exist.");
            }
            if (!string.IsNullOrEmpty(Model))
            {
                _modelFilepathOnnx = await OpenCVEnv.GetFilePathTaskAsync(Model, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_modelFilepathOnnx)) Debug.Log("The file:" + Model + " did not exist.");
#if OPENCV_SENTIS_AVAILABLE
                string sentisModelFileName = StreamingAssetPathOnnxToSentisIfNeeded(Model);
                _modelFilepathSentis = await OpenCVEnv.GetFilePathTaskAsync(
                    sentisModelFileName,
                    cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_modelFilepathSentis)) Debug.Log("The file:" + sentisModelFileName + " did not exist.");
#endif
            }

            if (_fpsMonitor != null)
                _fpsMonitor.ConsoleText = "";

            Run();
        }

        protected virtual void Run()
        {
            OpenCVDebug.SetDebugMode(true);

            InitializeInference();

            _multiSource2MatHelper.Initialize();
        }

        public virtual void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = _multiSource2MatHelper.GetMat();

            _texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);
            OpenCVMatUtils.MatToTexture2D(rgbaMat, _texture);

            ResultPreview.texture = _texture;
            ResultPreview.GetComponent<AspectRatioFitter>().aspectRatio = (float)_texture.width / _texture.height;

            if (_fpsMonitor != null)
            {
                _fpsMonitor.Add("width", rgbaMat.width().ToString());
                _fpsMonitor.Add("height", rgbaMat.height().ToString());
                _fpsMonitor.Add("orientation", Screen.orientation.ToString());
                UpdateFpsMonitorInferenceInfo(_fpsMonitor, _poseEstimator, UseAsyncInference);
            }

            _bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
        }

        public virtual void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            try
            {
                _poseEstimator?.Cancel();
            }
            catch (ObjectDisposedException)
            {
            }

            _inferenceRunner?.Cancel();

            _bgrMat?.Dispose(); _bgrMat = null;

            if (_texture != null) Texture2D.Destroy(_texture); _texture = null;
        }

        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode errorCode, string message)
        {
            Debug.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);

            if (_fpsMonitor != null)
            {
                _fpsMonitor.ConsoleText = "ErrorCode: " + errorCode + ":" + message;
            }
        }

        private void Update()
        {
            if (_inferenceReinitializing)
                return;

            if (_multiSource2MatHelper.IsPlaying() && _multiSource2MatHelper.DidUpdateThisFrame())
            {
                Mat rgbaMat = _multiSource2MatHelper.GetMat();

                if (_poseEstimator != null)
                {
                    Imgproc.cvtColor(rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);

                    if (_inferenceRunner != null)
                    {
                        _inferenceRunner.SubmitWork(
                            _bgrMat,
                            syncWork: m => _poseEstimator.Estimate(m, useCopyOutput: true),
                            asyncWork: async m =>
                            {
                                CancellationToken ct = _inferenceRunner.InFlightAsyncWorkCancellationToken;
                                return await _poseEstimator.EstimateTaskAsync(m, ct);
                            });

                        if (_inferenceRunner.TryGetLatestResult(out Mat detectedObjects))
                        {
                            _poseEstimator.Visualize(rgbaMat, detectedObjects, false, true);
                        }
                    }
                }

                OpenCVMatUtils.MatToTexture2D(rgbaMat, _texture);
            }
        }

        private async void OnDestroy()
        {
            _multiSource2MatHelper?.Dispose();

            await DisposeInferenceAsync();

            OpenCVDebug.SetDebugMode(false);

            _cts?.Dispose();
        }

        public virtual void OnBackButtonClick()
        {
            SceneManager.LoadScene("YOLOv8WithOpenCVForUnityExample");
        }

        public virtual void OnPlayButtonClick()
        {
            _multiSource2MatHelper.Play();
        }

        public virtual void OnPauseButtonClick()
        {
            _multiSource2MatHelper.Pause();
        }

        public virtual void OnStopButtonClick()
        {
            _multiSource2MatHelper.Stop();
        }

        public virtual void OnChangeCameraButtonClick()
        {
            _multiSource2MatHelper.RequestedIsFrontFacing = !_multiSource2MatHelper.RequestedIsFrontFacing;
        }

        /// <summary>
        /// Invoke from <c>UseSentisInferenceToggle</c> On Value Changed. Switches the inference backend.
        /// No-op when <c>OPENCV_SENTIS_AVAILABLE</c> is not defined.
        /// </summary>
        public async void OnUseSentisInferenceToggleValueChanged()
        {
#if !OPENCV_SENTIS_AVAILABLE
            await Task.CompletedTask;
            return;
#else
            if (UseSentisInferenceToggle == null || _inferenceReinitializing)
                return;

            bool newSentis = UseSentisInferenceToggle.isOn;
            if (newSentis == UseSentisInference)
                return;

            _inferenceReinitializing = true;
            UpdateInferenceModeToggles(inferenceReinitializing: true);

            await DisposeInferenceAsync();

            UseSentisInference = newSentis;
            UpdateUseAsyncInference();

            InitializeInference();

            UpdateFpsMonitorInferenceInfo(_fpsMonitor, _poseEstimator, UseAsyncInference);

            _inferenceReinitializing = false;
            UpdateInferenceModeToggles(inferenceReinitializing: false);
#endif
        }

        /// <summary>
        /// Invoke from <c>SentisBackendDropdown</c> On Value Changed. Switches Sentis backend type and reinitializes inference.
        /// No-op when <c>OPENCV_SENTIS_AVAILABLE</c> is not defined.
        /// </summary>
        public async void OnSentisBackendDropdownValueChanged(int index)
        {
#if !OPENCV_SENTIS_AVAILABLE
            await Task.CompletedTask;
            return;
#else
            if (SentisBackendDropdown == null || _inferenceReinitializing)
                return;

            int n = SentisBackendTypesInEnumOrder.Length;
            if (n == 0)
                return;
            int maxIdx = Mathf.Min(SentisBackendDropdown.options.Count, n) - 1;
            if (maxIdx < 0)
                return;
            BackendType newBackend = SentisBackendTypesInEnumOrder[Mathf.Clamp(index, 0, maxIdx)];
            if (newBackend == YoloSentisBackendType)
                return;

            _inferenceReinitializing = true;
            UpdateInferenceModeToggles(inferenceReinitializing: true);

            await DisposeInferenceAsync();

            YoloSentisBackendType = newBackend;
            UpdateUseSentisInference();
            UpdateUseAsyncInference();

            InitializeInference();

            UpdateFpsMonitorInferenceInfo(_fpsMonitor, _poseEstimator, UseAsyncInference);

            _inferenceReinitializing = false;
            UpdateInferenceModeToggles(inferenceReinitializing: false);
#endif
        }

        public void OnUseAsyncInferenceToggleValueChanged()
        {
            if (_inferenceReinitializing)
                return;
            if (UseAsyncInferenceToggle == null)
                return;
            if (UseAsyncInferenceToggle.isOn != UseAsyncInference)
            {
                if (_inferenceRunner != null)
                    _inferenceRunner.UseAsyncWork = UseAsyncInferenceToggle.isOn;
                UseAsyncInference = UseAsyncInferenceToggle.isOn;
                UpdateFpsMonitorInferenceInfo(_fpsMonitor, _poseEstimator, UseAsyncInference);
            }
        }

        private void UpdateInferenceModeToggles(bool inferenceReinitializing)
        {
            if (inferenceReinitializing)
            {
                if (UseSentisInferenceToggle != null)
                    UseSentisInferenceToggle.interactable = false;
                if (SentisBackendDropdown != null)
                    SentisBackendDropdown.interactable = false;
                if (UseAsyncInferenceToggle != null)
                    UseAsyncInferenceToggle.interactable = false;
                return;
            }

            if (UseAsyncInferenceToggle != null)
            {
                UseAsyncInferenceToggle.SetIsOnWithoutNotify(UseAsyncInference);
                UseAsyncInferenceToggle.interactable = true;
            }
#if OPENCV_SENTIS_AVAILABLE
            if (UseSentisInferenceToggle != null)
            {
                UseSentisInferenceToggle.SetIsOnWithoutNotify(UseSentisInference);
                UseSentisInferenceToggle.interactable = true;
            }
            if (SentisBackendDropdown != null)
                SentisBackendDropdown.interactable = UseSentisInference;
            UpdateSentisBackendDropdown();
#else
            if (UseSentisInferenceToggle != null)
            {
                UseSentisInferenceToggle.SetIsOnWithoutNotify(false);
                UseSentisInferenceToggle.interactable = false;
            }
            if (SentisBackendDropdown != null)
                SentisBackendDropdown.interactable = false;
#endif
        }

#if OPENCV_SENTIS_AVAILABLE
        private void UpdateSentisBackendDropdown()
        {
            if (SentisBackendDropdown == null || SentisBackendDropdown.options.Count == 0)
                return;
            if (SentisBackendTypesInEnumOrder.Length == 0)
                return;
            int idx = Array.IndexOf(SentisBackendTypesInEnumOrder, YoloSentisBackendType);
            if (idx < 0)
                idx = 0;
            int maxIdx = Mathf.Min(SentisBackendDropdown.options.Count, SentisBackendTypesInEnumOrder.Length) - 1;
            SentisBackendDropdown.SetValueWithoutNotify(Mathf.Clamp(idx, 0, maxIdx));
        }
#endif

        private void UpdateUseSentisInference()
        {
#if OPENCV_SENTIS_AVAILABLE
            if (!SystemInfo.supportsComputeShaders && YoloSentisBackendType == BackendType.GPUCompute)
                YoloSentisBackendType = BackendType.GPUPixel;
#endif
        }

        private void UpdateUseAsyncInference()
        {
        }

        private async Task DisposeInferenceAsync()
        {
            if (_inferenceRunner != null)
                await _inferenceRunner.DisposeAsync();
            _inferenceRunner = null;

            _poseEstimator?.Dispose();
            _poseEstimator = null;
        }

        private void InitializeInference()
        {
            string modelPath = _modelFilepathOnnx;
#if OPENCV_SENTIS_AVAILABLE
            if (UseSentisInference)
                modelPath = _modelFilepathSentis;
#endif
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("model: " + Model + " or " + "classes: " + Classes + " is not loaded. Please use [Tools] > [OpenCV for Unity] > [Setup Tools] > [Example Assets Downloader] to download the asset files required for this example scene, and then move them to the \"Assets/StreamingAssets\" folder.");
                if (_fpsMonitor != null)
                {
                    _fpsMonitor.Toast("model file is not loaded.\nPlease read console message.", 20000);
                }
                return;
            }

            try
            {
#if OPENCV_SENTIS_AVAILABLE
                if (UseSentisInference)
                {
                    _poseEstimator = new YOLOv8PoseEstimater(
                        modelPath,
                        _classesFilepath,
                        new Size(InpWidth, InpHeight),
                        ConfThreshold,
                        NmsThreshold,
                        TopK,
                        MultiBackendDnn.DNN_BACKEND_UNITY_SENTIS,
                        (int)YoloSentisBackendType);
                    Debug.Log("YOLOv8PoseEstimater initialized (Sentis / DNN_BACKEND_UNITY_SENTIS, backend=" + YoloSentisBackendType + ").");
                }
                else
#endif
                {
                    _poseEstimator = new YOLOv8PoseEstimater(
                        modelPath,
                        _classesFilepath,
                        new Size(InpWidth, InpHeight),
                        ConfThreshold,
                        NmsThreshold,
                        TopK);
                    Debug.Log("YOLOv8PoseEstimater initialized (OpenCV DNN).");
                }

                _inferenceRunner = new MatSingleFlightSyncAsyncRunner(
                    useAsyncWork: UseAsyncInference,
                    asyncWorkCancellationToken: _cts.Token,
                    disposeAsyncAfterWorkTask: async () =>
                    {
                        await _poseEstimator.WaitForCompletionTaskAsync();
                    });
            }
            catch (Exception ex)
            {
                Debug.LogWarning("YOLOv8PoseEstimationExample InitializeInference failed: " + ex);
            }
        }

        private static void UpdateFpsMonitorInferenceInfo(FpsMonitor fpsMonitor, DnnInferenceWorkerBase worker, bool useAsyncInference)
        {
            if (fpsMonitor == null)
                return;

            if (worker != null)
            {
                int be = worker.DnnBackend;
                int tgt = worker.DnnTarget;
                fpsMonitor.Add("dnnBackend", MultiBackendDnn.GetBackendDisplayString(be));
                fpsMonitor.Add("dnnTarget", MultiBackendDnn.GetTargetDisplayString(tgt));
            }
            else
            {
                fpsMonitor.Add("dnnBackend", "-");
                fpsMonitor.Add("dnnTarget", "-");
            }

            string useAsyncText = worker != null
                ? useAsyncInference.ToString()
                : "-";
            fpsMonitor.Add("useAsyncInference", useAsyncText);
        }

#if OPENCV_SENTIS_AVAILABLE
        private static string StreamingAssetPathOnnxToSentisIfNeeded(string streamingAssetsRelativePath)
        {
            if (string.IsNullOrEmpty(streamingAssetsRelativePath))
                return streamingAssetsRelativePath;
            if (!streamingAssetsRelativePath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                return streamingAssetsRelativePath;
            return Path.ChangeExtension(streamingAssetsRelativePath, ".sentis");
        }
#endif
    }
}

#endif
