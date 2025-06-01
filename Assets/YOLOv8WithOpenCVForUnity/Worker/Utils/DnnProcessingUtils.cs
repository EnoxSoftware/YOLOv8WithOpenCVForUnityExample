using System;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.UnityUtils;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.Utils
{
    /// <summary>
    /// Utility class for OpenCV DNN processing operations.
    /// Provides common functionality for DNN-based processing, including:
    /// - Non-maximum suppression (NMS) implementations
    /// - Result processing and transformations
    /// - Common DNN operations
    /// </summary>
    public static class DnnProcessingUtils
    {
        /// <summary>
        /// Performs class-wise non-maximum suppression using the Ultralytics approach.
        /// This method processes detections by:
        /// 1. Adding class-specific offsets to bounding boxes
        /// 2. Applying NMS on all boxes at once
        /// 3. Returning the indices of the kept detections
        /// </summary>
        /// <param name="bboxes">
        /// a set of bounding boxes to apply NMS.
        /// </param>
        /// <param name="scores">
        /// a set of corresponding confidences.
        /// </param>
        /// <param name="class_ids">
        /// a set of corresponding class ids. Ids are integer and usually start from 0.
        /// </param>
        /// <param name="score_threshold">
        /// a threshold used to filter boxes by score.
        /// </param>
        /// <param name="nms_threshold">
        /// a threshold used in non maximum suppression.
        /// </param>
        /// <param name="indices">
        /// the kept indices of bboxes after NMS.
        /// </param>
        /// <param name="eta">
        /// a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
        /// </param>
        /// <param name="top_k">
        /// if `&gt;0`, keep at most @p top_k picked indices.
        /// </param>
        /// <param name="max_wh">
        /// Maximum box width and height in pixels.
        /// </param>
        public static void NMSBoxesClassWise(MatOfRect2d bboxes, MatOfFloat scores, MatOfInt class_ids, float score_threshold,
                                                 float nms_threshold, MatOfInt indices, float eta, int top_k, int max_wh = 7680)
        {
            if (bboxes != null) bboxes.ThrowIfDisposed();
            if (scores != null) scores.ThrowIfDisposed();
            if (class_ids != null) class_ids.ThrowIfDisposed();
            if (indices != null) indices.ThrowIfDisposed();

            if (bboxes.total() != scores.total())
                throw new ArgumentException("bboxes and scores must have the same number of elements");
            if (scores.total() != class_ids.total())
                throw new ArgumentException("scores and class_ids must have the same number of elements");

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            ReadOnlySpan<int> allClassIds = class_ids.AsSpan<int>();
            ReadOnlySpan<Vec4d> allBBoxes = bboxes.AsSpan<Vec4d>();
#else
            int[] allClassIds = class_ids.toArray();
            Vec4d[] allBBoxes = bboxes.toVec4dArray();
#endif

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            using (Mat offsetBBoxes = bboxes.clone())
            {
                Span<Vec4d> allOffsetBBoxes = offsetBBoxes.AsSpan<Vec4d>();

                for (int i = 0; i < allBBoxes.Length; i++)
                {
                    double offset = allClassIds[i] * max_wh;

                    allOffsetBBoxes[i].Item1 = allBBoxes[i].Item1 + offset;
                    allOffsetBBoxes[i].Item2 = allBBoxes[i].Item2 + offset;
                }

                using (MatOfRect2d offsetBBoxesMat = new MatOfRect2d(offsetBBoxes))
                {
                    Dnn.NMSBoxes(offsetBBoxesMat, scores, score_threshold, nms_threshold, indices, eta, top_k);
                }
            }
#else
            Vec4d[] allOffsetBBoxes = allBBoxes.Clone() as Vec4d[];

            for (int i = 0; i < allBBoxes.Length; i++)
            {
                double offset = allClassIds[i] * max_wh;

                allOffsetBBoxes[i].Item1 = allBBoxes[i].Item1 + offset;
                allOffsetBBoxes[i].Item2 = allBBoxes[i].Item2 + offset;
            }

            using (MatOfRect2d offsetBBoxesMat = new MatOfRect2d(allOffsetBBoxes))
            {
                Dnn.NMSBoxes(offsetBBoxesMat, scores, score_threshold, nms_threshold, indices, eta, top_k);
            }
#endif
        }
    }
}