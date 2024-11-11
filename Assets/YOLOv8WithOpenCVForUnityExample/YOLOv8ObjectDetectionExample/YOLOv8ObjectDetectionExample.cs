#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

#if !UNITY_WSA_10_0

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using YOLOv8WithOpenCVForUnity;
using System.Threading;
using UnityEngine.UI;

namespace YOLOv8WithOpenCVForUnityExample
{
    /// <summary>
    /// YOLOv8 Object Detection Example
    /// Referring to https://github.com/ultralytics/ultralytics/
    /// https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
    /// </summary>
    [RequireComponent(typeof(MultiSource2MatHelper))]
    public class YOLOv8ObjectDetectionExample : MonoBehaviour
    {
        [Header("Output")]
        /// <summary>
        /// The RawImage for previewing the result.
        /// </summary>
        public RawImage resultPreview;

        [Space(10)]

        [TooltipAttribute("Path to a binary file of model contains trained weights.")]
        public string model = "yolov8n.onnx";

        [TooltipAttribute("Optional path to a text file with names of classes to label detected objects.")]
        public string classes = "coco.names";

        [TooltipAttribute("Confidence threshold.")]
        public float confThreshold = 0.25f;

        [TooltipAttribute("Non-maximum suppression threshold.")]
        public float nmsThreshold = 0.45f;

        [TooltipAttribute("Maximum detections per image.")]
        public int topK = 300;

        [TooltipAttribute("Preprocess input image by resizing to a specific width.")]
        public int inpWidth = 640;

        [TooltipAttribute("Preprocess input image by resizing to a specific height.")]
        public int inpHeight = 640;

        /// <summary>
        /// The texture.
        /// </summary>
        protected Texture2D texture;

        /// <summary>
        /// The multi source to mat helper.
        /// </summary>
        protected MultiSource2MatHelper multiSource2MatHelper;

        /// <summary>
        /// The bgr mat.
        /// </summary>
        protected Mat bgrMat;

        /// <summary>
        /// The YOLOv8 ObjectDetector.
        /// </summary>
        YOLOv8ObjectDetector objectDetector;

        /// <summary>
        /// The FPS monitor.
        /// </summary>
        protected FpsMonitor fpsMonitor;

        protected string classes_filepath;
        protected string model_filepath;

        /// <summary>
        /// The CancellationTokenSource.
        /// </summary>
        CancellationTokenSource cts = new CancellationTokenSource();

        // Use this for initialization
        async void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();

            multiSource2MatHelper = gameObject.GetComponent<MultiSource2MatHelper>();
            multiSource2MatHelper.outputColorFormat = Source2MatHelperColorFormat.RGBA;

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            if (fpsMonitor != null)
                fpsMonitor.consoleText = "Preparing file access...";

            if (!string.IsNullOrEmpty(classes))
                classes_filepath = await Utils.getFilePathAsyncTask("YOLOv8WithOpenCVForUnityExample/" + classes, cancellationToken: cts.Token);
            if (!string.IsNullOrEmpty(model))
                model_filepath = await Utils.getFilePathAsyncTask("YOLOv8WithOpenCVForUnityExample/" + model, cancellationToken: cts.Token);

            if (fpsMonitor != null)
                fpsMonitor.consoleText = "";

            Run();
        }

        // Use this for initialization
        protected virtual void Run()
        {
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            Utils.setDebugMode(true);

            if (string.IsNullOrEmpty(model_filepath))
            {
                Debug.LogError("model: " + model + " is not loaded.");
            }
            else
            {
                objectDetector = new YOLOv8ObjectDetector(model_filepath, classes_filepath, new Size(inpWidth, inpHeight), confThreshold, nmsThreshold, topK);
            }

            multiSource2MatHelper.Initialize();
        }

        /// <summary>
        /// Raises the source to mat helper initialized event.
        /// </summary>
        public virtual void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = multiSource2MatHelper.GetMat();

            texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);

            resultPreview.texture = texture;
            resultPreview.GetComponent<AspectRatioFitter>().aspectRatio = (float)texture.width / texture.height;


            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", rgbaMat.width().ToString());
                fpsMonitor.Add("height", rgbaMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }

            bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public virtual void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            if (bgrMat != null)
                bgrMat.Dispose();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }
        }

        /// <summary>
        /// Raises the source to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        /// <param name="message">Message.</param>
        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode errorCode, string message)
        {
            Debug.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);

            if (fpsMonitor != null)
            {
                fpsMonitor.consoleText = "ErrorCode: " + errorCode + ":" + message;
            }
        }

        // Update is called once per frame
        protected virtual void Update()
        {
            if (multiSource2MatHelper.IsPlaying() && multiSource2MatHelper.DidUpdateThisFrame())
            {

                Mat rgbaMat = multiSource2MatHelper.GetMat();

                if (objectDetector == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {

                    Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR);

                    //TickMeter tm = new TickMeter();
                    //tm.start();

                    Mat results = objectDetector.infer(bgrMat);

                    //tm.stop();
                    //Debug.Log("YOLOv8ObjectDetector Inference time (preprocess + infer + postprocess), ms: " + tm.getTimeMilli());

                    Imgproc.cvtColor(bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);

                    objectDetector.visualize(rgbaMat, results, false, true);

                }

                Utils.matToTexture2D(rgbaMat, texture);
            }
        }

        /// <summary>
        /// Raises the destroy event.
        /// </summary>
        protected virtual void OnDestroy()
        {
            multiSource2MatHelper.Dispose();

            if (objectDetector != null)
                objectDetector.dispose();

            Utils.setDebugMode(false);

            if (cts != null)
                cts.Dispose();
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
            multiSource2MatHelper.Play();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public virtual void OnPauseButtonClick()
        {
            multiSource2MatHelper.Pause();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public virtual void OnStopButtonClick()
        {
            multiSource2MatHelper.Stop();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public virtual void OnChangeCameraButtonClick()
        {
            multiSource2MatHelper.requestedIsFrontFacing = !multiSource2MatHelper.requestedIsFrontFacing;
        }
    }
}
#endif

#endif