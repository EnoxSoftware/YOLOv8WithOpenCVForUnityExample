using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using OpenCVRange = OpenCVForUnity.CoreModule.Range;
using OpenCVRect = OpenCVForUnity.CoreModule.Rect;

namespace YOLOv8WithOpenCVForUnity
{
    /// <summary>
    /// Referring to https://github.com/ultralytics/ultralytics/
    /// https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
    /// </summary>
    public class YOLOv8ObjectDetector : IDisposable
    {
        Size input_size;
        float conf_threshold;
        float nms_threshold;
        int topK;
        int backend;
        int target;

        int num_classes = 80;
        bool class_agnostic = false;// Non-use of multi-class NMS

        Net object_detection_net;
        List<string> classNames;

        List<Scalar> palette;

        Mat paddedImg;

        Mat pickup_blob_numx6;
        Mat boxesMat;

        Mat boxes_m_c4;
        Mat confidences_m;
        Mat class_ids_m;
        MatOfRect2d boxes;
        MatOfFloat confidences;
        MatOfInt class_ids;

        public YOLOv8ObjectDetector(string modelFilepath, string classesFilepath, Size inputSize, float confThreshold = 0.25f, float nmsThreshold = 0.45f, int topK = 300, int backend = Dnn.DNN_BACKEND_OPENCV, int target = Dnn.DNN_TARGET_CPU)
        {
            // initialize
            if (!string.IsNullOrEmpty(modelFilepath))
            {
                object_detection_net = Dnn.readNet(modelFilepath);
            }

            if (!string.IsNullOrEmpty(classesFilepath))
            {
                classNames = ReadClassNames(classesFilepath);
                num_classes = classNames.Count;
            }

            input_size = new Size(inputSize.width > 0 ? inputSize.width : 640, inputSize.height > 0 ? inputSize.height : 640);
            conf_threshold = Mathf.Clamp01(confThreshold);
            nms_threshold = Mathf.Clamp01(nmsThreshold);
            this.topK = topK;
            this.backend = backend;
            this.target = target;

            object_detection_net.setPreferableBackend(this.backend);
            object_detection_net.setPreferableTarget(this.target);

            palette = new List<Scalar>();
            palette.Add(new Scalar(255, 56, 56, 255));
            palette.Add(new Scalar(255, 157, 151, 255));
            palette.Add(new Scalar(255, 112, 31, 255));
            palette.Add(new Scalar(255, 178, 29, 255));
            palette.Add(new Scalar(207, 210, 49, 255));
            palette.Add(new Scalar(72, 249, 10, 255));
            palette.Add(new Scalar(146, 204, 23, 255));
            palette.Add(new Scalar(61, 219, 134, 255));
            palette.Add(new Scalar(26, 147, 52, 255));
            palette.Add(new Scalar(0, 212, 187, 255));
            palette.Add(new Scalar(44, 153, 168, 255));
            palette.Add(new Scalar(0, 194, 255, 255));
            palette.Add(new Scalar(52, 69, 147, 255));
            palette.Add(new Scalar(100, 115, 255, 255));
            palette.Add(new Scalar(0, 24, 236, 255));
            palette.Add(new Scalar(132, 56, 255, 255));
            palette.Add(new Scalar(82, 0, 133, 255));
            palette.Add(new Scalar(203, 56, 255, 255));
            palette.Add(new Scalar(255, 149, 200, 255));
            palette.Add(new Scalar(255, 55, 199, 255));
        }

        protected virtual Mat PreProcess(Mat image)
        {
            // https://github.com/ultralytics/ultralytics/blob/d74a5a9499acf1afd13d970645e5b1cfcadf4a8f/ultralytics/data/augment.py#L645

            // Add padding to make it input size.
            // (padding to center the image)
            float ratio = Mathf.Max((float)image.cols() / (float)input_size.width, (float)image.rows() / (float)input_size.height);
            int padw = (int)Mathf.Ceil((float)input_size.width * ratio);
            int padh = (int)Mathf.Ceil((float)input_size.height * ratio);

            if (paddedImg == null)
                paddedImg = new Mat(padh, padw, image.type(), Scalar.all(114));
            if (paddedImg.width() != padw || paddedImg.height() != padh)
            {
                paddedImg.create(padh, padw, image.type());
                Imgproc.rectangle(paddedImg, new OpenCVRect(0, 0, paddedImg.width(), paddedImg.height()), Scalar.all(114), -1);
            }

            Mat _paddedImg_roi = new Mat(paddedImg, new OpenCVRect((paddedImg.cols() - image.cols()) / 2, (paddedImg.rows() - image.rows()) / 2, image.cols(), image.rows()));
            image.copyTo(_paddedImg_roi);

            Mat blob = Dnn.blobFromImage(paddedImg, 1.0 / 255.0, input_size, Scalar.all(0), true, false, CvType.CV_32F); // HWC to NCHW, BGR to RGB

            return blob;// [1, 3, h, w]
        }

        public virtual Mat Infer(Mat image)
        {
            // cheack
            if (image.channels() != 3)
            {
                Debug.Log("The input image must be in BGR format.");
                return new Mat();
            }

            // Preprocess
            Mat input_blob = PreProcess(image);

            // Forward
            object_detection_net.setInput(input_blob);

            List<Mat> output_blob = new List<Mat>();
            object_detection_net.forward(output_blob, object_detection_net.getUnconnectedOutLayersNames());

            // Postprocess
            Mat results = PostProcess(output_blob[0], image.size());

            // scale_boxes
            float ratio = Mathf.Max((float)image.cols() / (float)input_size.width, (float)image.rows() / (float)input_size.height);
            float x_factor = ratio;
            float y_factor = ratio;
            float x_shift = ((float)input_size.width * ratio - (float)image.size().width) / 2f;
            float y_shift = ((float)input_size.height * ratio - (float)image.size().height) / 2f;

            for (int i = 0; i < results.rows(); ++i)
            {
                float[] results_arr = new float[4];
                results.get(i, 0, results_arr);
                float x1 = Mathf.Round(results_arr[0] * x_factor - x_shift);
                float y1 = Mathf.Round(results_arr[1] * y_factor - y_shift);
                float x2 = Mathf.Round(results_arr[2] * x_factor - x_shift);
                float y2 = Mathf.Round(results_arr[3] * y_factor - y_shift);

                results.put(i, 0, new float[] { x1, y1, x2, y2 });
            }


            input_blob.Dispose();
            for (int i = 0; i < output_blob.Count; i++)
            {
                output_blob[i].Dispose();
            }

            return results;// [n, 6] (xyxy, conf, cls)
        }

        protected virtual Mat PostProcess(Mat output_blob, Size original_shape)
        {
            Mat output_blob_0 = output_blob;

            // 1*84*8400 -> 1*8400*84
            MatOfInt order = new MatOfInt(0, 2, 1);
            Core.transposeND(output_blob_0, order, output_blob_0);

            if (output_blob_0.size(2) != 4 + num_classes)
            {
                Debug.LogWarning("The number of classes and output shapes are different. " +
                                "( output_blob_0.size(2):" + output_blob_0.size(2) + " != 4 + num_classes:" + num_classes + " )\n" +
                                "When using a custom model, be sure to set the correct number of classes by loading the appropriate custom classesFile.");

                num_classes = output_blob_0.size(2) - 4;
            }

            int num = output_blob_0.size(1);
            Mat output_blob_numx84 = output_blob_0.reshape(1, num);
            Mat box_delta = output_blob_numx84.colRange(new OpenCVRange(0, 4));
            Mat classes_scores_delta = output_blob_numx84.colRange(new OpenCVRange(4, 4 + num_classes));

            // pre-NMS
            // Pick up rows to process by conf_threshold value and calculate scores and class_ids.
            if (pickup_blob_numx6 == null)
                pickup_blob_numx6 = new Mat(300, 6, CvType.CV_32FC1, new Scalar(0));

            Imgproc.rectangle(pickup_blob_numx6, new OpenCVRect(4, 0, 1, pickup_blob_numx6.rows()), Scalar.all(0), -1);

            int ind = 0;
            for (int i = 0; i < num; ++i)
            {
                Mat cls_scores = classes_scores_delta.row(i);
                Core.MinMaxLocResult minmax = Core.minMaxLoc(cls_scores);
                float conf = (float)minmax.maxVal;

                if (conf > conf_threshold)
                {
                    if (ind > pickup_blob_numx6.rows())
                    {
                        Mat _conf_blob_numx6 = new Mat(pickup_blob_numx6.rows() * 2, pickup_blob_numx6.cols(), pickup_blob_numx6.type(), new Scalar(0));
                        pickup_blob_numx6.copyTo(_conf_blob_numx6.rowRange(0, pickup_blob_numx6.rows()));
                        pickup_blob_numx6 = _conf_blob_numx6;
                    }

                    float[] box_arr = new float[4];
                    box_delta.get(i, 0, box_arr);

                    pickup_blob_numx6.put(ind, 0, new float[] { box_arr[0], box_arr[1], box_arr[2], box_arr[3], conf, (float)minmax.maxLoc.x });
                    ind++;
                }
            }

            int num_pickup = pickup_blob_numx6.rows();
            Mat pickup_box_delta = pickup_blob_numx6.colRange(new OpenCVRange(0, 4));
            Mat pickup_confidence = pickup_blob_numx6.colRange(new OpenCVRange(4, 5));

            // Convert boxes from [cx, cy, w, h] to [x, y, w, h] where Rect2d data style.
            if (boxesMat == null || boxesMat.rows() != num_pickup)
                boxesMat = new Mat(num_pickup, 4, CvType.CV_32FC1);
            Mat cxy_delta = pickup_box_delta.colRange(new OpenCVRange(0, 2));
            Mat wh_delta = pickup_box_delta.colRange(new OpenCVRange(2, 4));
            Mat xy1 = boxesMat.colRange(new OpenCVRange(0, 2));
            Mat xy2 = boxesMat.colRange(new OpenCVRange(2, 4));
            wh_delta.copyTo(xy2);
            Core.divide(wh_delta, new Scalar(2.0), wh_delta);
            Core.subtract(cxy_delta, wh_delta, xy1);


            if (boxes_m_c4 == null || boxes_m_c4.rows() != num_pickup)
                boxes_m_c4 = new Mat(num_pickup, 1, CvType.CV_64FC4);
            if (confidences_m == null || confidences_m.rows() != num_pickup)
                confidences_m = new Mat(num_pickup, 1, CvType.CV_32FC1);

            if (boxes == null || boxes.rows() != num_pickup)
                boxes = new MatOfRect2d(boxes_m_c4);
            if (confidences == null || confidences.rows() != num_pickup)
                confidences = new MatOfFloat(confidences_m);


            // non-maximum suppression
            Mat boxes_m_c1 = boxes_m_c4.reshape(1, num_pickup);
            boxesMat.convertTo(boxes_m_c1, CvType.CV_64F);
            pickup_confidence.copyTo(confidences_m);

            MatOfInt indices = new MatOfInt();

            if (class_agnostic)
            {
                // NMS
                Dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices, 1f, topK);
            }
            else
            {
                Mat pickup_class_ids = pickup_blob_numx6.colRange(new OpenCVRange(5, 6));

                if (class_ids_m == null || class_ids_m.rows() != num_pickup)
                    class_ids_m = new Mat(num_pickup, 1, CvType.CV_32SC1);
                if (class_ids == null || class_ids.rows() != num_pickup)
                    class_ids = new MatOfInt(class_ids_m);

                pickup_class_ids.convertTo(class_ids_m, CvType.CV_32S);

                // multi-class NMS
                Dnn.NMSBoxesBatched(boxes, confidences, class_ids, conf_threshold, nms_threshold, indices, 1f, topK);
            }

            Mat results = new Mat(indices.rows(), 6, CvType.CV_32FC1);

            for (int i = 0; i < indices.rows(); ++i)
            {
                int idx = (int)indices.get(i, 0)[0];

                pickup_blob_numx6.row(idx).copyTo(results.row(i));

                float[] bbox_arr = new float[4];
                boxesMat.get(idx, 0, bbox_arr);
                float x = bbox_arr[0];
                float y = bbox_arr[1];
                float w = bbox_arr[2];
                float h = bbox_arr[3];
                results.put(i, 0, new float[] { x, y, x + w, y + h });
            }

            indices.Dispose();

            // [
            //   [xyxy, conf, cls]
            //   ...
            //   [xyxy, conf, cls]
            // ]
            return results;

        }

        public virtual void Visualize(Mat image, Mat results, bool print_results = false, bool isRGB = false)
        {
            if (image.IsDisposed)
                return;

            if (results.empty() || results.cols() < 6)
                return;

            DetectionData[] data = GetData(results);

            foreach (var d in data.Reverse())
            {
                float left = d.x1;
                float top = d.y1;
                float right = d.x2;
                float bottom = d.y2;
                float conf = d.conf;
                int classId = (int)d.cls;

                Scalar c = palette[classId % palette.Count];
                Scalar color = isRGB ? c : new Scalar(c.val[2], c.val[1], c.val[0], c.val[3]);

                Imgproc.rectangle(image, new Point(left, top), new Point(right, bottom), color, 2);

                string label = $"{GetClassLabel(classId)}, {conf:F2}";

                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

                top = Mathf.Max((float)top, (float)labelSize.height);
                Imgproc.rectangle(image, new Point(left, top - labelSize.height),
                    new Point(left + labelSize.width, top + baseLine[0]), color, Core.FILLED);
                Imgproc.putText(image, label, new Point(left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(255), 1, Imgproc.LINE_AA);
            }

            // Print results
            if (print_results)
            {
                StringBuilder sb = new StringBuilder(512);

                for (int i = 0; i < data.Length; ++i)
                {
                    var d = data[i];
                    string label = GetClassLabel(d.cls);

                    sb.AppendFormat("-----------object {0}-----------", i + 1);
                    sb.AppendLine();
                    sb.AppendFormat("conf: {0:F4}", d.conf);
                    sb.AppendLine();
                    sb.Append("cls: ").Append(label);
                    sb.AppendLine();
                    sb.AppendFormat("box: {0:F0} {1:F0} {2:F0} {3:F0}", d.x1, d.y1, d.x2, d.y2);
                    sb.AppendLine();
                }

                Debug.Log(sb.ToString());
            }
        }

        public virtual void Dispose()
        {
            if (object_detection_net != null)
                object_detection_net.Dispose();

            if (paddedImg != null)
                paddedImg.Dispose();

            paddedImg = null;

            if (pickup_blob_numx6 != null)
                pickup_blob_numx6.Dispose();
            if (boxesMat != null)
                boxesMat.Dispose();

            pickup_blob_numx6 = null;
            boxesMat = null;

            if (boxes_m_c4 != null)
                boxes_m_c4.Dispose();
            if (confidences_m != null)
                confidences_m.Dispose();
            if (class_ids_m != null)
                class_ids_m.Dispose();
            if (boxes != null)
                boxes.Dispose();
            if (confidences != null)
                confidences.Dispose();
            if (class_ids != null)
                class_ids.Dispose();

            boxes_m_c4 = null;
            confidences_m = null;
            class_ids_m = null;
            boxes = null;
            confidences = null;
            class_ids = null;
        }

        protected virtual List<string> ReadClassNames(string filename)
        {
            List<string> classNames = new List<string>();

            System.IO.StreamReader cReader = null;
            try
            {
                cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

                while (cReader.Peek() >= 0)
                {
                    string name = cReader.ReadLine();
                    classNames.Add(name);
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError(ex.Message);
                return null;
            }
            finally
            {
                if (cReader != null)
                    cReader.Close();
            }

            return classNames;
        }

        [Serializable]
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct DetectionData
        {
            public readonly float x1;
            public readonly float y1;
            public readonly float x2;
            public readonly float y2;
            public readonly float conf;
            public readonly float cls;

            // Count of elements
            public const int ELEMENT_COUNT = 6;

            // sizeof(DetectionData)
            public const int DATA_SIZE = ELEMENT_COUNT * sizeof(float);

            public DetectionData(int x1, int y1, int x2, int y2, float conf, int cls)
            {
                this.x1 = x1;
                this.y1 = y1;
                this.x2 = x2;
                this.y2 = y2;
                this.conf = conf;
                this.cls = cls;
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder(128);
                sb.AppendFormat("x1:{0} y1:{1} x2:{2} y2:{3} conf:{4} cls:{5}", x1, y1, x2, y2, conf, cls);
                return sb.ToString();
            }
        };

        public virtual DetectionData[] GetData(Mat results)
        {
            if (results.empty())
                return new DetectionData[0];

            var dst = new DetectionData[results.rows()];
            MatUtils.copyFromMat(results, dst);

            return dst;
        }

        public virtual string GetClassLabel(float id)
        {
            int classId = (int)id;
            string className = string.Empty;
            if (classNames != null && classNames.Count != 0)
            {
                if (classId >= 0 && classId < classNames.Count)
                {
                    className = classNames[classId];
                }
            }
            if (string.IsNullOrEmpty(className))
                className = classId.ToString();

            return className;
        }
    }
}