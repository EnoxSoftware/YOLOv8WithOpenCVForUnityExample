using System;
using System.Collections.Generic;
using OpenCVForUnity.CoreModule;
using UnityEngine;

namespace YOLOv8WithOpenCVForUnity.UnityIntegration.Worker.Utils
{
    /// <summary>
    /// A pool for managing and reusing Mat objects to improve performance and reduce memory allocation.
    /// This pool is designed to handle Mat objects of a specific size and type.
    /// </summary>
    public class MatPool : IDisposable
    {
        private readonly Queue<Mat> _pool;
        private readonly Size _size;
        private readonly int _type;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MatPool class.
        /// </summary>
        /// <param name="size">The size of Mat objects to be managed by this pool.</param>
        /// <param name="type">The type of Mat objects to be managed by this pool.</param>
        /// <param name="initialPoolSize">The initial number of Mat objects to create in the pool. Default is 1.</param>
        public MatPool(Size size, int type, int initialPoolSize = 1)
        {
            _pool = new Queue<Mat>();
            _size = size;
            _type = type;
            _disposed = false;

            // Create initial Mat objects for the pool
            for (int i = 0; i < initialPoolSize; i++)
            {
                _pool.Enqueue(new Mat(_size, _type));
            }
        }

        /// <summary>
        /// Gets a Mat object from the pool. If the pool is empty, creates a new Mat object.
        /// </summary>
        /// <returns>A Mat object from the pool or a newly created one.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the pool has been disposed.</exception>
        public Mat Get()
        {
            ThrowIfDisposed();

            if (_pool.Count > 0)
            {
                return _pool.Dequeue();
            }

            // If pool is empty, create a new Mat
            return new Mat(_size, _type);
        }

        /// <summary>
        /// Returns a Mat object to the pool. If the Mat has different size or type than the pool specification,
        /// it will be released instead of being returned to the pool.
        /// </summary>
        /// <param name="mat">The Mat object to return to the pool.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the pool has been disposed.</exception>
        public void Return(Mat mat)
        {
            ThrowIfDisposed();

            if (mat == null) return;

            // Check if the Mat has the correct size and type
            if (mat.size() != _size || mat.type() != _type)
            {
                Debug.LogWarning("Returned Mat has different size or type than pool specification");
                mat.Dispose();
                return;
            }

            _pool.Enqueue(mat);
        }

        /// <summary>
        /// Disposes all Mat objects in the pool and resets the pool state.
        /// This method should be called when the pool is no longer needed.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of Dispose pattern.
        /// </summary>
        /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;

            if (disposing)
            {
                while (_pool.Count > 0)
                {
                    var mat = _pool.Dequeue();
                    if (mat != null)
                    {
                        mat.Dispose();
                    }
                }
            }

            _disposed = true;
        }

        /// <summary>
        /// Throws an ObjectDisposedException if the pool has been disposed.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown when the pool has been disposed.</exception>
        protected void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(MatPool));
            }
        }

        /// <summary>
        /// Finalizer for MatPool.
        /// </summary>
        ~MatPool()
        {
            Dispose(false);
        }
    }
}