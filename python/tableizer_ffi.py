"""
Python FFI bindings for the Tableizer C++ library.
Provides access to table detection and coordinate transformation functions.
"""

import ctypes
from ctypes import c_char_p, c_int, c_float, POINTER, c_uint8
import json
import numpy as np
import platform
import os
from pathlib import Path
import cv2
import base64


class TableizerFFI:
    """Python wrapper for Tableizer C++ library FFI functions."""

    def __init__(self, library_path=None):
        """
        Initialize the FFI wrapper with the C++ library.

        Parameters
        ----------
        library_path : str, optional
            Path to the compiled C++ library. If None, will try to find it automatically.
        """
        self.lib = None
        self._load_library(library_path)
        self._setup_functions()

    def _load_library(self, library_path=None):
        """Load the C++ library."""
        if library_path is None:
            # Try to find the library automatically
            library_path = self._find_library()

        if not os.path.exists(library_path):
            raise FileNotFoundError(f"Tableizer library not found at: {library_path}")

        try:
            self.lib = ctypes.CDLL(library_path)
            print(f"Loaded Tableizer library from: {library_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Tableizer library: {e}")

    def _find_library(self):
        """Automatically find the compiled library based on platform."""
        # Start from the python directory and look for the library
        base_path = Path(__file__).parent.parent

        if platform.system() == "Darwin":  # macOS
            lib_patterns = [
                "lib/build/libtableizer_lib.dylib",
                "lib/build/libtableizer.dylib",
                "lib/build/Debug/libtableizer_lib.dylib",
                "lib/build/Debug/libtableizer.dylib",
                "lib/build/Release/libtableizer_lib.dylib",
                "lib/build/Release/libtableizer.dylib",
                "build/libtableizer_lib.dylib",
                "build/libtableizer.dylib",
            ]
        elif platform.system() == "Linux":
            lib_patterns = [
                "lib/build/libtableizer.so",
                "lib/build/Debug/libtableizer.so",
                "lib/build/Release/libtableizer.so",
                "build/libtableizer.so",
            ]
        elif platform.system() == "Windows":
            lib_patterns = [
                "lib/build/tableizer.dll",
                "lib/build/Debug/tableizer.dll",
                "lib/build/Release/tableizer.dll",
                "build/tableizer.dll",
            ]
        else:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")

        for pattern in lib_patterns:
            lib_path = base_path / pattern
            if lib_path.exists():
                return str(lib_path)

        raise FileNotFoundError(
            f"Could not find Tableizer library. Searched patterns: {lib_patterns}"
        )

    def _setup_functions(self):
        """Setup function signatures for the C++ FFI functions."""
        # Table detection function (BGRA version)
        self.lib.detect_table_bgra.argtypes = [
            POINTER(c_uint8),  # image_bytes
            c_int,  # width
            c_int,  # height
            c_int,  # stride
            c_int,  # rotation_degrees
            c_char_p,  # debug_image_path
        ]
        self.lib.detect_table_bgra.restype = c_char_p

        # Coordinate transformation function
        self.lib.transform_points_using_quad.argtypes = [
            POINTER(c_float),  # points_data
            c_int,  # points_count
            POINTER(c_float),  # quad_data
            c_int,  # quad_count
            c_int,  # image_width
            c_int,  # image_height
            c_int,  # display_width
            c_int,  # display_height
        ]
        self.lib.transform_points_using_quad.restype = c_char_p

    def detect_table(self, image, rotation_degrees=0, debug_path=None):
        """
        Detect table quad points in an image using C++ implementation.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        rotation_degrees : int
            Rotation in degrees (0, 90, 180, 270) - not used in RGBA version
        debug_path : str, optional
            Path to save debug image - not used in RGBA version

        Returns
        -------
        dict or None
            Dictionary containing:
            - 'quad_points': List of 4 quad points as [[x, y], [x, y], ...]
            - 'mask': numpy.ndarray (grayscale) if available, None otherwise
            - 'image': base64 encoded debug image
            Returns None if detection failed
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")

        # Convert BGR to BGRA for the BGRA function
        bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        height, width = bgra_image.shape[:2]
        stride = bgra_image.strides[0]  # Use OpenCV stride

        # Convert to ctypes array
        image_data = bgra_image.astype(np.uint8)
        image_ptr = image_data.ctypes.data_as(POINTER(c_uint8))

        try:
            # Call C++ BGRA function
            debug_path_ptr = None  # No debug output for now
            result_ptr = self.lib.detect_table_bgra(
                image_ptr,
                c_int(width),
                c_int(height),
                c_int(stride),
                c_int(rotation_degrees),  # rotation_degrees parameter
                debug_path_ptr,  # debug_image_path parameter
            )

            if not result_ptr:
                return None

            # Convert result to Python string
            result_json = result_ptr.decode("utf-8")
            result_data = json.loads(result_json)

            if "error" in result_data:
                print(f"Table detection error: {result_data['error']}")
                return None

            # Process mask data if present
            if "mask" in result_data:
                mask_data = result_data["mask"]
                try:
                    # Decode base64 mask back to numpy array
                    mask_bytes = base64.b64decode(mask_data["data"])
                    mask_array = cv2.imdecode(
                        np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
                    )
                    result_data["mask"] = mask_array
                except Exception as e:
                    print(f"Warning: Failed to decode mask data: {e}")
                    result_data["mask"] = None

            return result_data

        except Exception as e:
            print(f"Error in table detection: {e}")
            return None

    def transform_points(self, points, quad_points, image_size, display_size):
        """
        Transform points using quad-to-rectangle perspective transformation.

        Parameters
        ----------
        points : list or np.ndarray
            List of points to transform, each as [x, y] or (x, y)
        quad_points : list
            List of 4 quad points as [x, y] pairs
        image_size : tuple
            (width, height) of the source image
        display_size : tuple
            (width, height) of the destination display

        Returns
        -------
        list or None
            List of transformed points as [{"x": float, "y": float}, ...] or None if failed
        """
        if len(quad_points) != 4:
            raise ValueError("quad_points must contain exactly 4 points")

        if len(points) == 0:
            return []

        # Convert points to flat array [x1, y1, x2, y2, ...]
        points_array = np.array(points, dtype=np.float32).flatten()
        if len(points_array) % 2 != 0:
            raise ValueError("Points must be pairs of (x, y) coordinates")

        points_count = len(points_array) // 2

        # Convert quad points to flat array [x1, y1, x2, y2, x3, y3, x4, y4]
        quad_array = np.array(quad_points, dtype=np.float32).flatten()
        if len(quad_array) != 8:
            raise ValueError("quad_points must contain exactly 4 (x, y) pairs")

        # Create ctypes pointers
        points_ptr = points_array.ctypes.data_as(POINTER(c_float))
        quad_ptr = quad_array.ctypes.data_as(POINTER(c_float))

        try:
            # Call C++ function
            result_ptr = self.lib.transform_points_using_quad(
                points_ptr,
                c_int(points_count),
                quad_ptr,
                c_int(4),
                c_int(image_size[0]),  # width
                c_int(image_size[1]),  # height
                c_int(display_size[0]),  # width
                c_int(display_size[1]),  # height
            )

            if not result_ptr:
                return None

            # Convert result to Python string
            result_json = result_ptr.decode("utf-8")
            result_data = json.loads(result_json)

            if "error" in result_data:
                print(f"Transform points error: {result_data['error']}")
                return None

            return result_data.get("transformed_points", [])

        except Exception as e:
            print(f"Error in point transformation: {e}")
            return None


# Global instance for easy access
_tableizer_ffi = None


def get_tableizer_ffi():
    """Get the global TableizerFFI instance, creating it if necessary."""
    global _tableizer_ffi
    if _tableizer_ffi is None:
        _tableizer_ffi = TableizerFFI()
    return _tableizer_ffi


# Convenience functions
def detect_table_cpp(image, rotation_degrees=0, debug_path=None):
    """
    Convenience function for table detection using C++ implementation.

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'quad_points': List of 4 corner points as [[x, y], [x, y], ...]
        - 'mask': numpy.ndarray (grayscale mask) if available, None otherwise
        - 'image': base64 encoded debug image
        Returns None if detection failed
    """
    ffi = get_tableizer_ffi()
    return ffi.detect_table(image, rotation_degrees, debug_path)


def transform_points_cpp(points, quad_points, image_size, display_size):
    """Convenience function for point transformation using C++ implementation."""
    ffi = get_tableizer_ffi()
    return ffi.transform_points(points, quad_points, image_size, display_size)


if __name__ == "__main__":
    # Test the FFI bindings
    print("Testing Tableizer FFI bindings...")

    try:
        ffi = TableizerFFI()
        print("✅ FFI initialization successful")

        # Test with a simple image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:380, 160:480] = [0, 128, 0]  # Green rectangle

        quad_points = ffi.detect_table(test_image)
        if quad_points:
            print(f"✅ Table detection successful: {len(quad_points)} points found")

            # Test point transformation
            test_points = [[320, 240], [300, 200]]  # Center and offset points
            transformed = ffi.transform_points(
                test_points,
                [(pt["x"], pt["y"]) for pt in quad_points],
                (640, 480),
                (400, 600),
            )

            if transformed:
                print(
                    f"✅ Point transformation successful: {len(transformed)} points transformed"
                )
            else:
                print("❌ Point transformation failed")
        else:
            print("❌ Table detection failed")

    except Exception as e:
        print(f"❌ FFI test failed: {e}")
