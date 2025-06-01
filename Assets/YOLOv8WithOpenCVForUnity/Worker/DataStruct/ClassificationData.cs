using System;
using System.Runtime.InteropServices;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.DataStruct
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public readonly struct ClassificationData
    {
        public readonly float Confidence;
        private readonly float _rawClassId;

        public readonly int ClassId => (int)_rawClassId;

        public const int ELEMENT_COUNT = 2;
        public static readonly int DATA_SIZE = ELEMENT_COUNT * Marshal.SizeOf<float>();

        public ClassificationData(float confidence, int classId)
        {
            Confidence = confidence;
            _rawClassId = classId;
        }

        public readonly override string ToString()
        {
            return $"ClassificationData(Confidence:{Confidence}, ClassId:{ClassId})";
        }
    }
}
