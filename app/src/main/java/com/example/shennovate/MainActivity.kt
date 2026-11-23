package com.example.shennovate

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private lateinit var accelText: TextView
    private lateinit var gyroText: TextView
    private lateinit var statusText: TextView
    private lateinit var predictionText: TextView
    private lateinit var activitySpinner: Spinner
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var recordInfoText: TextView

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private val accelerationWindow = mutableListOf<Float>()
    private val xWindow = mutableListOf<Float>()
    private val yWindow = mutableListOf<Float>()
    private val zWindow = mutableListOf<Float>()
    private val gyroWindow = mutableListOf<Float>()
    private val windowSize = 50
    private var latestGyroMagnitude = 0f

    private var lastAccelX = 0f
    private var lastAccelY = 0f
    private var lastAccelZ = 0f
    private var lastGyroX = 0f
    private var lastGyroY = 0f
    private var lastGyroZ = 0f

    private var isRecording = false
    private var currentFile: File? = null
    private var recordedRows = 0

    private val jerkSpikeThreshold = 4.010694951443163f

    private val labels = listOf(
        "Stationary",
        "Walking",
        "Running",
        "Sudden stop",
        "Normal phone movement"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        accelText = findViewById(R.id.accelText)
        gyroText = findViewById(R.id.gyroText)
        statusText = findViewById(R.id.statusText)
        predictionText = findViewById(R.id.predictionText)
        activitySpinner = findViewById(R.id.activitySpinner)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        recordInfoText = findViewById(R.id.recordInfoText)

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        activitySpinner.adapter = adapter

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        initOnnxModel()

        startButton.setOnClickListener {
            startRecording()
        }

        stopButton.setOnClickListener {
            stopRecording()
        }
    }

    private fun initOnnxModel() {
        try {
            val modelBytes = assets.open("cadence_model.onnx").readBytes()
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv?.createSession(modelBytes)
            predictionText.text = "Cadence ML: Ready"
        } catch (e: Exception) {
            predictionText.text = "Cadence ML: Model load error"
        }
    }

    private fun startRecording() {
        val selectedActivity = activitySpinner.selectedItem.toString()
        val cleanActivity = selectedActivity.lowercase(Locale.US).replace(" ", "_")
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

        val folder = getExternalFilesDir(null) ?: filesDir
        val file = File(folder, "cadence_${cleanActivity}_$timeStamp.csv")

        file.writeText("Timestamp,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,Activity\n")

        currentFile = file
        recordedRows = 0
        isRecording = true

        startButton.isEnabled = false
        stopButton.isEnabled = true
        activitySpinner.isEnabled = false

        recordInfoText.text = "Recording ($selectedActivity)..."
    }

    private fun stopRecording() {
        isRecording = false
        startButton.isEnabled = true
        stopButton.isEnabled = false
        activitySpinner.isEnabled = true

        val savedPath = currentFile?.name ?: "file"
        recordInfoText.text = "Saved $recordedRows rows to $savedPath"
    }

    private fun saveRowIfRecording() {
        if (!isRecording) return
        val file = currentFile ?: return

        val now = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
        val activityName = activitySpinner.selectedItem.toString()

        val line = "$now,$lastAccelX,$lastAccelY,$lastAccelZ,$lastGyroX,$lastGyroY,$lastGyroZ,$activityName\n"
        file.appendText(line)
        recordedRows++

        recordInfoText.text = "Recording ($activityName): $recordedRows rows saved"
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        if (isRecording) {
            stopRecording()
        }
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        ortSession?.close()
        ortEnv?.close()
    }

    private fun calculateStdDev(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average().toFloat()
        val variance = values.map { (it - mean).pow(2) }.average().toFloat()
        return sqrt(variance)
    }

    private fun buildFeatureVector17(): FloatArray {
        val accelAvg = accelerationWindow.average().toFloat()
        val accelPeak = accelerationWindow.maxOrNull() ?: 0f
        val accelMin = accelerationWindow.minOrNull() ?: 0f
        val accelRange = accelPeak - accelMin
        val accelBounce = calculateStdDev(accelerationWindow)

        val accelJerks = if (accelerationWindow.size > 1) {
            accelerationWindow.zipWithNext { a, b -> abs(b - a) }
        } else {
            listOf(0f)
        }
        val accelJerkMax = accelJerks.maxOrNull() ?: 0f
        val accelJerkAvg = accelJerks.average().toFloat()

        val halfIndex = accelerationWindow.size / 2
        val firstHalfAvg = if (halfIndex > 0) {
            accelerationWindow.subList(0, halfIndex).average().toFloat()
        } else {
            accelAvg
        }
        val secondHalfAvg = if (accelerationWindow.size - halfIndex > 0) {
            accelerationWindow.subList(halfIndex, accelerationWindow.size).average().toFloat()
        } else {
            accelAvg
        }
        val halfDrop = firstHalfAvg - secondHalfAvg

        val gyroAvg = gyroWindow.average().toFloat()
        val gyroPeak = gyroWindow.maxOrNull() ?: 0f
        val gyroMin = gyroWindow.minOrNull() ?: 0f
        val gyroRange = gyroPeak - gyroMin
        val gyroBounce = calculateStdDev(gyroWindow)

        val gyroJerks = if (gyroWindow.size > 1) {
            gyroWindow.zipWithNext { a, b -> abs(b - a) }
        } else {
            listOf(0f)
        }
        val gyroJerkMax = gyroJerks.maxOrNull() ?: 0f

        val accelGyroRatio = accelAvg / gyroAvg.coerceAtLeast(1e-4f)

        val xStd = calculateStdDev(xWindow)
        val yStd = calculateStdDev(yWindow)
        val zStd = calculateStdDev(zWindow)

        // Exact feature order 0..16
        return floatArrayOf(
            accelAvg,         // 0: accel_avg
            accelBounce,      // 1: accel_bounce
            accelPeak,        // 2: accel_peak
            accelMin,         // 3: accel_min
            accelRange,       // 4: accel_range
            accelJerkMax,     // 5: accel_jerk_max
            accelJerkAvg,     // 6: accel_jerk_avg
            halfDrop,         // 7: half_drop
            gyroAvg,          // 8: gyro_avg
            gyroBounce,       // 9: gyro_bounce
            gyroPeak,         // 10: gyro_peak
            gyroRange,        // 11: gyro_range
            gyroJerkMax,      // 12: gyro_jerk_max
            accelGyroRatio,   // 13: accel_gyro_ratio
            xStd,             // 14: x_std
            yStd,             // 15: y_std
            zStd              // 16: z_std
        )
    }

    private fun runOnnxPrediction(features17: FloatArray) {
        val session = ortSession ?: return
        val env = ortEnv ?: return

        try {
            val inputName = session.inputNames.iterator().next()
            val tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(features17),
                longArrayOf(1L, features17.size.toLong())
            )
            tensor.use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { results ->
                    val rawOutput = results[0].value
                    val predictedLabel = when (rawOutput) {
                        is Array<*> -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        is LongArray -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        is FloatArray -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        else -> rawOutput.toString()
                    }
                    val spikeFlag = if (features17[5] >= jerkSpikeThreshold) " [SPIKE]" else ""
                    predictionText.text = "Cadence ML: $predictedLabel$spikeFlag"
                }
            }
        } catch (e: Exception) {
            predictionText.text = "Cadence ML: Inference error"
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            lastAccelX = x
            lastAccelY = y
            lastAccelZ = z

            val magnitude = sqrt(x * x + y * y + z * z)
            val dynamicAcceleration = abs(magnitude - 9.81f)

            accelerationWindow.add(dynamicAcceleration)
            xWindow.add(x)
            yWindow.add(y)
            zWindow.add(z)

            if (accelerationWindow.size > windowSize) accelerationWindow.removeAt(0)
            if (xWindow.size > windowSize) xWindow.removeAt(0)
            if (yWindow.size > windowSize) yWindow.removeAt(0)
            if (zWindow.size > windowSize) zWindow.removeAt(0)

            val variation = calculateStdDev(accelerationWindow)

            accelText.text = String.format(
                "X: %.2f\nY: %.2f\nZ: %.2f\nMagnitude: %.2f",
                x, y, z, magnitude
            )

            if (accelerationWindow.size < windowSize) {
                statusText.text = "Collecting motion data..."
            } else {
                statusText.text = if (variation > 2.0f) {
                    "High motion variation"
                } else {
                    "Low motion variation"
                }
                val features = buildFeatureVector17()
                runOnnxPrediction(features)
            }

            saveRowIfRecording()
        }

        if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            lastGyroX = x
            lastGyroY = y
            lastGyroZ = z

            latestGyroMagnitude = sqrt(x * x + y * y + z * z)
            gyroWindow.add(latestGyroMagnitude)
            if (gyroWindow.size > windowSize) gyroWindow.removeAt(0)

            gyroText.text = String.format(
                "X: %.2f\nY: %.2f\nZ: %.2f\nRotation: %.2f",
                x, y, z, latestGyroMagnitude
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed
    }
}