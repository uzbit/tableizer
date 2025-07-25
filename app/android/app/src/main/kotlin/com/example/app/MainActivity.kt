package com.example.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class MainActivity : FlutterActivity() {
    private val channel = "com.example.tableizer/detect"
    private var module: Module? = null

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channel).setMethodCallHandler { call, result ->
            if (call.method == "detect") {
                val args = call.arguments as Map<String, Any>
                val width = args["width"] as Int
                val height = args["height"] as Int
                val planes = args["planes"] as List<ByteArray>

                val yBuffer = planes[0]
                val uBuffer = planes[1]
                val vBuffer = planes[2]

                val yuvImage = YuvImage(yBuffer, ImageFormat.YUV_420_888, width, height, null)
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
                val imageBytes = out.toByteArray()
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                if (module == null) {
                    module = LiteModuleLoader.load(assetFilePath("detection_model.pte"))
                }

                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB
                )

                val outputTensor = module!!.forward(inputTensor).toTensor()
                val scores = outputTensor.dataAsFloatArray

                // TODO: Process the output tensor to extract ball data.
                // This is a placeholder and needs to be adapted to the model's output format.
                val balls = mutableListOf<Map<String, Any>>()
                for (i in scores.indices step 6) {
                    val score = scores[i]
                    if (score > 0.5) { // Confidence threshold
                        val x = scores[i + 1] * width
                        val y = scores[i + 2] * height
                        val radius = scores[i + 3] * width
                        val classId = scores[i + 5].toInt()
                        balls.add(mapOf("x" to x, "y" to y, "radius" to radius, "class_id" to classId))
                    }
                }
                result.success(balls)
            } else {
                result.notImplemented()
            }
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = java.io.File(cacheDir, assetName)
        if (!file.exists()) {
            assets.open(assetName).use { `is` ->
                file.outputStream().use { os ->
                    `is`.copyTo(os)
                }
            }
        }
        return file.absolutePath
    }
}
