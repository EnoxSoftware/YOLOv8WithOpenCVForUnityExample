using System;
using System.Runtime.InteropServices;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.DataStruct
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public readonly struct ObjectDetectionData
    {
        public readonly float X1;
        public readonly float Y1;
        public readonly float X2;
        public readonly float Y2;
        public readonly float Confidence;
        private readonly float _rawClassId;

        public readonly int ClassId => (int)_rawClassId;

        public const int ELEMENT_COUNT = 6;
        public static readonly int DATA_SIZE = ELEMENT_COUNT * Marshal.SizeOf<float>();

        public ObjectDetectionData(float x1, float y1, float x2, float y2, float confidence, int classId)
        {
            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
            Confidence = confidence;
            _rawClassId = classId;
        }

        public readonly override string ToString()
        {
            return $"DetectionData(X1:{X1} Y1:{Y1} X2:{X2} Y2:{Y2} Confidence:{Confidence} ClassId:{ClassId})";
        }
    }
}
