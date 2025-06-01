using System;
using System.Runtime.InteropServices;
using System.Text;
using OpenCVForUnity.UnityUtils;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.DataStruct
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public unsafe struct PoseEstimationCOCO17DataUnsafe
    {
        public readonly float X1;
        public readonly float Y1;
        public readonly float X2;
        public readonly float Y2;
        public readonly float Confidence;
        private readonly float _rawClassId;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = LANDMARK_ELEMENT_COUNT)]
        public fixed float _rawLandmarks[LANDMARK_ELEMENT_COUNT];

        public readonly int ClassId => (int)_rawClassId;
        public readonly Vec3f Nose => new Vec3f(_rawLandmarks[0], _rawLandmarks[1], _rawLandmarks[2]);
        public readonly Vec3f LeftEye => new Vec3f(_rawLandmarks[3], _rawLandmarks[4], _rawLandmarks[5]);
        public readonly Vec3f RightEye => new Vec3f(_rawLandmarks[6], _rawLandmarks[7], _rawLandmarks[8]);
        public readonly Vec3f LeftEar => new Vec3f(_rawLandmarks[9], _rawLandmarks[10], _rawLandmarks[11]);
        public readonly Vec3f RightEar => new Vec3f(_rawLandmarks[12], _rawLandmarks[13], _rawLandmarks[14]);
        public readonly Vec3f LeftShoulder => new Vec3f(_rawLandmarks[15], _rawLandmarks[16], _rawLandmarks[17]);
        public readonly Vec3f RightShoulder => new Vec3f(_rawLandmarks[18], _rawLandmarks[19], _rawLandmarks[20]);
        public readonly Vec3f LeftElbow => new Vec3f(_rawLandmarks[21], _rawLandmarks[22], _rawLandmarks[23]);
        public readonly Vec3f RightElbow => new Vec3f(_rawLandmarks[24], _rawLandmarks[25], _rawLandmarks[26]);
        public readonly Vec3f LeftWrist => new Vec3f(_rawLandmarks[27], _rawLandmarks[28], _rawLandmarks[29]);
        public readonly Vec3f RightWrist => new Vec3f(_rawLandmarks[30], _rawLandmarks[31], _rawLandmarks[32]);
        public readonly Vec3f LeftHip => new Vec3f(_rawLandmarks[33], _rawLandmarks[34], _rawLandmarks[35]);
        public readonly Vec3f RightHip => new Vec3f(_rawLandmarks[36], _rawLandmarks[37], _rawLandmarks[38]);
        public readonly Vec3f LeftKnee => new Vec3f(_rawLandmarks[39], _rawLandmarks[40], _rawLandmarks[41]);
        public readonly Vec3f RightKnee => new Vec3f(_rawLandmarks[42], _rawLandmarks[43], _rawLandmarks[44]);
        public readonly Vec3f LeftAnkle => new Vec3f(_rawLandmarks[45], _rawLandmarks[46], _rawLandmarks[47]);
        public readonly Vec3f RightAnkle => new Vec3f(_rawLandmarks[48], _rawLandmarks[49], _rawLandmarks[50]);

        public const int LANDMARK_VEC3F_COUNT = 17;
        public const int LANDMARK_ELEMENT_COUNT = 3 * LANDMARK_VEC3F_COUNT;
        public const int ELEMENT_COUNT = 6 + LANDMARK_ELEMENT_COUNT;
        public static readonly int DATA_SIZE = ELEMENT_COUNT * Marshal.SizeOf<float>();

        public PoseEstimationCOCO17DataUnsafe(float x1, float y1, float x2, float y2, float confidence, int classId, Vec3f[] landmarks)
        {
            if (landmarks == null || landmarks.Length != LANDMARK_VEC3F_COUNT)
                throw new ArgumentException("landmarks must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");

            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
            Confidence = confidence;
            _rawClassId = classId;
            fixed (Vec3f* src = landmarks)
            fixed (float* dest = _rawLandmarks)
            {
                Buffer.MemoryCopy(src, dest, LANDMARK_ELEMENT_COUNT * sizeof(float), LANDMARK_ELEMENT_COUNT * sizeof(float));
            }
        }

        public readonly ReadOnlySpan<Vec3f> GetLandmarks()
        {
            fixed (float* ptr = _rawLandmarks)
            {
                return new ReadOnlySpan<Vec3f>(ptr, LANDMARK_VEC3F_COUNT);
            }
        }

        public readonly override string ToString()
        {
            StringBuilder sb = new StringBuilder(512);

            sb.Append("PoseEstimationCOCO17DataUnsafe(");
            sb.AppendFormat("X1:{0} Y1:{1} X2:{2} Y2:{3} Confidence:{4} ClassId:{5} ", X1, Y1, X2, Y2, Confidence, ClassId);
            ReadOnlySpan<Vec3f> landmarks = GetLandmarks();
            sb.Append("Landmarks:");
            foreach (var p in landmarks)
            {
                sb.Append(p.ToString());
            }
            sb.Append(")");

            return sb.ToString();
        }
    }
}
