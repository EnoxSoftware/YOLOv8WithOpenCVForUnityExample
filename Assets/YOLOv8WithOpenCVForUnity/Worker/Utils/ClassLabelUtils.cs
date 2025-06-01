using System;
using System.Collections.Generic;
using UnityEngine;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.Utils
{
    /// <summary>
    /// Utility class for handling class labels and names.
    /// </summary>
    public static class ClassLabelUtils
    {
        /// <summary>
        /// Reads class names from a text file.
        /// </summary>
        /// <param name="filename">Path to the text file containing class names.</param>
        /// <returns>List of class names.</returns>
        /// <exception cref="ArgumentNullException">Thrown when filename is null or empty.</exception>
        /// <exception cref="System.IO.FileNotFoundException">Thrown when the file does not exist.</exception>
        public static List<string> ReadClassNames(string filename)
        {
            if (string.IsNullOrEmpty(filename))
                throw new ArgumentNullException(nameof(filename), "Filename cannot be null or empty.");

            if (!System.IO.File.Exists(filename))
                throw new System.IO.FileNotFoundException($"Class names file not found: {filename}");

            List<string> classNames = new List<string>();

            System.IO.StreamReader cReader = null;
            try
            {
                cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

                while (cReader.Peek() >= 0)
                {
                    string name = cReader.ReadLine();
                    if (!string.IsNullOrEmpty(name))
                    {
                        classNames.Add(name);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error reading class names file: {ex.Message}");
                throw;
            }
            finally
            {
                if (cReader != null)
                    cReader.Close();
            }

            return classNames;
        }

        /// <summary>
        /// Gets the class label for the given class ID.
        /// </summary>
        /// <param name="id">Class ID.</param>
        /// <param name="classNames">List of class names.</param>
        /// <returns>Class label string. Returns the ID as string if no label is found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when classNames is null.</exception>
        public static string GetClassLabel(float id, List<string> classNames)
        {
            if (classNames == null)
                throw new ArgumentNullException(nameof(classNames), "Class names list cannot be null.");

            int classId = (int)id;
            if (classId < 0)
                return classId.ToString();

            if (classNames.Count == 0)
                return classId.ToString();

            if (classId >= classNames.Count)
                return classId.ToString();

            string className = classNames[classId];
            return string.IsNullOrEmpty(className) ? classId.ToString() : className;
        }
    }
}