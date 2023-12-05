using UnityEngine;
using UnityEngine.SceneManagement;

namespace YOLOv8WithOpenCVForUnityExample
{

    public class ShowLicense : MonoBehaviour
    {

        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("YOLOv8WithOpenCVForUnityExample");
        }
    }
}
