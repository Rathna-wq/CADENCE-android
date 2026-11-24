package com.example.shennovate

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.toColorInt
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    enum class SafetyState {
        GREEN_MONITORING,
        YELLOW_LISTENING_HELP,
        RED_GRACE_PERIOD,
        ALERT_DISPATCHED
    }

    enum class MicCueMode(val label: String) {
        HELP_KEYWORD_SHOUT("Help keyword + loud voice"),
        LOUD_SCREAM("Loud noise / shout detection"),
        TOO_MUCH_SILENCE("Way too much silence")
    }

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var locationManager: LocationManager? = null

    private lateinit var accelText: TextView
    private lateinit var gyroText: TextView
    private lateinit var micModeSpinner: Spinner
    private lateinit var micText: TextView
    private lateinit var micButton: Button
    private lateinit var statusText: TextView
    private lateinit var predictionText: TextView
    private lateinit var resetButton: Button
    private lateinit var alertLogText: TextView

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private val accelerationWindow = ArrayDeque<Float>()
    private val xWindow = ArrayDeque<Float>()
    private val yWindow = ArrayDeque<Float>()
    private val zWindow = ArrayDeque<Float>()
    private val gyroWindow = ArrayDeque<Float>()
    private val windowSize = 50
    private var latestGyroMagnitude = 0f

    private val mainHandler = Handler(Looper.getMainLooper())

    private var selectedMicMode = MicCueMode.HELP_KEYWORD_SHOUT
    private var audioRecord: AudioRecord? = null
    @Volatile private var isMicMonitoring = false
    private var micThread: Thread? = null
    @Volatile private var latestAudioRms = 0f
    @Volatile private var peakAudioRms = 0f
    @Volatile private var loudStreakCount = 0
    @Volatile private var silenceStartMs: Long? = null

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListeningForHelp = false
    @Volatile private var helpKeywordDetected = false
    private var lastHelpHeardTranscript = ""

    private val loudAudioThreshold = 1500f
    private val shoutRmsThreshold = 1200f
    private val loudStreakRequired = 2
    private val silenceRmsThreshold = 250f
    private val excessiveSilenceDurationMs = 5000L

    private var lastVibrationMs = 0L
    private val vibrationCooldownMs = 6_000L

    private var latestLat: Double? = null
    private var latestLon: Double? = null
    private var latestSpeedMps: Float = 0f

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            latestLat = location.latitude
            latestLon = location.longitude
            if (location.hasSpeed()) {
                latestSpeedMps = location.speed
            }
        }
    }

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val micGranted = grants[Manifest.permission.RECORD_AUDIO] == true
            if (micGranted) {
                startMicMonitoring()
            } else {
                micText.text = getString(R.string.mic_permission_denied)
            }
        }

    private var lastEvalMs = 0L
    private val evalIntervalMs = 100L

    private val recentSpikes = ArrayDeque<Boolean>()
    private val historySize = 6

    private val jerkSpikeThreshold = 4.010695f
    private val suddenStopAccelJumpThreshold = 8.5f

    private var currentSafetyState = SafetyState.GREEN_MONITORING
    private var listeningStartMs: Long? = null
    private val maxHelpListenWindowMs = 12_000L
    private var graceStartMs: Long? = null
    private val gracePeriodDurationMs = 8_000L

    private var lastPredictedLabel = "Stationary"
    private var lastJerkMax = 0f
    private var lastTriggerReason = "init"
    private var statusDetailSuffix = ""
    private var alertSentForCurrentIncident = false

    private val graceCountdownRunnable = object : Runnable {
        override fun run() {
            if (currentSafetyState == SafetyState.RED_GRACE_PERIOD) {
                val now = System.currentTimeMillis()
                val elapsed = now - (graceStartMs ?: now)
                val remainingMs = (gracePeriodDurationMs - elapsed).coerceAtLeast(0L)
                val remainingSec = remainingMs / 1000.0

                if (remainingMs <= 0L) {
                    transitionToAlertDispatched("grace_period_expired")
                } else {
                    statusDetailSuffix = " (${String.format(Locale.US, "%.1f", remainingSec)}s — Tap Cancel to stop)"
                    applySafetyUi(currentSafetyState, statusDetailSuffix)
                    mainHandler.postDelayed(this, 200L)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        accelText = findViewById(R.id.accelText)
        gyroText = findViewById(R.id.gyroText)
        micModeSpinner = findViewById(R.id.micModeSpinner)
        micText = findViewById(R.id.micText)
        micButton = findViewById(R.id.micButton)
        statusText = findViewById(R.id.statusText)
        predictionText = findViewById(R.id.predictionText)
        resetButton = findViewById(R.id.resetButton)
        alertLogText = findViewById(R.id.alertLogText)

        resetButton.text = "Cancel Alert / Reset State"

        val micModeAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            MicCueMode.entries.map { it.label }
        )
        micModeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        micModeSpinner.adapter = micModeAdapter
        micModeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedMicMode = MicCueMode.entries[position]
                resetMicCueFlags()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        locationManager = getSystemService(LOCATION_SERVICE) as? LocationManager

        applySafetyUi(SafetyState.GREEN_MONITORING, "")
        initOnnxModel()
        initSpeechRecognizer()

        micButton.setOnClickListener {
            toggleMicMonitoring()
        }

        resetButton.setOnClickListener {
            cancelOrResetSafetyState()
        }
    }

    private fun initSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            return
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    if (isListeningForHelp && currentSafetyState == SafetyState.YELLOW_LISTENING_HELP) {
                        restartHelpSpeechListen()
                    }
                }

                override fun onError(error: Int) {
                    if (isListeningForHelp && currentSafetyState == SafetyState.YELLOW_LISTENING_HELP) {
                        mainHandler.postDelayed({
                            if (isListeningForHelp && currentSafetyState == SafetyState.YELLOW_LISTENING_HELP) {
                                restartHelpSpeechListen()
                            }
                        }, 350L)
                    }
                }

                override fun onResults(results: Bundle?) {
                    handleSpeechResults(results)
                    if (isListeningForHelp && currentSafetyState == SafetyState.YELLOW_LISTENING_HELP) {
                        restartHelpSpeechListen()
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    handleSpeechResults(partialResults)
                }

                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
        }
    }

    private fun handleSpeechResults(bundle: Bundle?) {
        val matches = bundle?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: return
        val combined = matches.joinToString(" ").lowercase(Locale.US)
        if (combined.isNotBlank()) {
            lastHelpHeardTranscript = combined
        }
        val containsHelp = Regex("\\b(help|halp|elp)\\b").containsMatchIn(combined)
        if (containsHelp) {
            helpKeywordDetected = true
            mainHandler.post {
                if (currentSafetyState == SafetyState.YELLOW_LISTENING_HELP) {
                    enterGracePeriod("keyword 'help' heard (\"$combined\")")
                }
            }
        }
    }

    private fun startHelpSpeechListening() {
        if (isListeningForHelp) return
        val hasMic = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (!hasMic) return

        helpKeywordDetected = false
        lastHelpHeardTranscript = ""
        isListeningForHelp = true
        restartHelpSpeechListen()
    }

    private fun restartHelpSpeechListen() {
        val recognizer = speechRecognizer ?: return
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
        }
        try {
            recognizer.cancel()
            recognizer.startListening(intent)
        } catch (_: Exception) {
        }
    }

    private fun stopHelpSpeechListening() {
        isListeningForHelp = false
        try {
            speechRecognizer?.stopListening()
            speechRecognizer?.cancel()
        } catch (_: Exception) {
        }
    }

    private fun triggerAlertVibration(force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && now - lastVibrationMs < vibrationCooldownMs) return
        lastVibrationMs = now

        mainHandler.post {
            try {
                val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                    vibratorManager.defaultVibrator
                } else {
                    @Suppress("DEPRECATION")
                    getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                }

                if (!vibrator.hasVibrator()) return@post

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val timings = longArrayOf(0, 400, 200, 400, 200, 600)
                    val amplitudes = intArrayOf(0, 220, 0, 220, 0, 255)
                    vibrator.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(1400L)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun resetMicCueFlags() {
        loudStreakCount = 0
        silenceStartMs = null
        helpKeywordDetected = false
    }

    private fun toggleMicMonitoring() {
        if (isMicMonitoring) {
            stopMicMonitoring()
        } else {
            val needed = mutableListOf<String>()
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                needed.add(Manifest.permission.RECORD_AUDIO)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                needed.add(Manifest.permission.ACCESS_FINE_LOCATION)
            }
            if (needed.isEmpty()) {
                startMicMonitoring()
            } else {
                requestPermissionsLauncher.launch(needed.toTypedArray())
            }
        }
    }

    private fun startLocationUpdates() {
        val lm = locationManager ?: return
        val fineGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        val coarseGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (!fineGranted && !coarseGranted) return

        try {
            if (fineGranted && lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1500L, 1f, locationListener)
            }
            if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                lm.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 2500L, 5f, locationListener)
            }
            val lastKnown = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            if (lastKnown != null) {
                latestLat = lastKnown.latitude
                latestLon = lastKnown.longitude
                if (lastKnown.hasSpeed()) latestSpeedMps = lastKnown.speed
            }
        } catch (_: SecurityException) {
        }
    }

    private fun stopLocationUpdates() {
        try {
            locationManager?.removeUpdates(locationListener)
        } catch (_: Exception) {
        }
    }

    private fun startMicMonitoring() {
        if (isMicMonitoring) return
        resetMicCueFlags()
        startLocationUpdates()
        isMicMonitoring = true
        micButton.text = getString(R.string.disable_microphone)

        val sampleRate = 16000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            micText.text = getString(R.string.mic_unsupported_buffer)
            isMicMonitoring = false
            micButton.text = getString(R.string.enable_microphone)
            return
        }

        try {
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuf * 2
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                micText.text = getString(R.string.mic_init_failed)
                isMicMonitoring = false
                micButton.text = getString(R.string.enable_microphone)
                return
            }

            audioRecord = recorder
            peakAudioRms = 0f
            micText.text = "Mic: Armed & monitoring audio..."

            recorder.startRecording()

            micThread = Thread {
                val buffer = ShortArray(minBuf)
                while (isMicMonitoring) {
                    val readCount = recorder.read(buffer, 0, buffer.size)
                    if (readCount > 0) {
                        var sumSq = 0.0
                        for (i in 0 until readCount) {
                            val s = buffer[i].toDouble()
                            sumSq += s * s
                        }
                        val rms = sqrt(sumSq / readCount).toFloat()
                        val now = System.currentTimeMillis()
                        latestAudioRms = rms
                        if (rms > peakAudioRms) {
                            peakAudioRms = rms
                        }

                        val isShoutDetected = rms >= shoutRmsThreshold || rms >= loudAudioThreshold
                        if (isShoutDetected) {
                            loudStreakCount++
                        } else {
                            loudStreakCount = 0
                        }

                        if (rms <= silenceRmsThreshold) {
                            if (silenceStartMs == null) {
                                silenceStartMs = now
                            }
                        } else {
                            silenceStartMs = null
                        }

                        val silenceElapsedMs = silenceStartMs?.let { now - it } ?: 0L
                        val isLoudEvent = loudStreakCount >= loudStreakRequired
                        val isExcessiveSilence = silenceElapsedMs >= excessiveSilenceDurationMs

                        mainHandler.post {
                            if (!isMicMonitoring) return@post

                            if (currentSafetyState == SafetyState.YELLOW_LISTENING_HELP &&
                                (helpKeywordDetected || (selectedMicMode == MicCueMode.HELP_KEYWORD_SHOUT && isLoudEvent))
                            ) {
                                val reason = if (helpKeywordDetected) {
                                    "keyword 'help' detected (rms=${rms.toInt()})"
                                } else {
                                    "loud shout during motion alert (rms=${rms.toInt()})"
                                }
                                enterGracePeriod(reason)
                            } else if (currentSafetyState == SafetyState.GREEN_MONITORING &&
                                selectedMicMode == MicCueMode.LOUD_SCREAM &&
                                isLoudEvent
                            ) {
                                enterYellowListeningState("loud scream detected (rms=${rms.toInt()})")
                            }

                            micText.text = when (selectedMicMode) {
                                MicCueMode.HELP_KEYWORD_SHOUT -> {
                                    val kwTag = if (helpKeywordDetected) " [HELP HEARD]" else ""
                                    val loudTag = if (isLoudEvent) " [LOUD]" else ""
                                    String.format(Locale.US, "RMS: %.0f | Peak: %.0f%s%s", rms, peakAudioRms, kwTag, loudTag)
                                }
                                MicCueMode.LOUD_SCREAM -> {
                                    val tag = if (isLoudEvent || isShoutDetected) " [SHOUT/LOUD DETECTED]" else ""
                                    String.format(Locale.US, "RMS: %.0f | Peak: %.0f%s", rms, peakAudioRms, tag)
                                }
                                MicCueMode.TOO_MUCH_SILENCE -> {
                                    val secSilent = silenceElapsedMs / 1000.0
                                    val tag = if (isExcessiveSilence) " [TOO SILENT]" else ""
                                    String.format(Locale.US, "RMS: %.0f | Silent: %.1fs%s", rms, secSilent, tag)
                                }
                            }
                        }
                    }
                }
            }.apply {
                isDaemon = true
                start()
            }
        } catch (_: SecurityException) {
            micText.text = getString(R.string.mic_permission_required)
            isMicMonitoring = false
            micButton.text = getString(R.string.enable_microphone)
        } catch (_: Exception) {
            micText.text = getString(R.string.mic_error_starting)
            isMicMonitoring = false
            micButton.text = getString(R.string.enable_microphone)
        }
    }

    private fun stopMicMonitoring() {
        isMicMonitoring = false
        stopHelpSpeechListening()
        try {
            micThread?.join(300)
        } catch (_: InterruptedException) {
        }
        micThread = null

        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {
        }
        audioRecord = null

        latestAudioRms = 0f
        resetMicCueFlags()
        micButton.text = getString(R.string.enable_microphone)
        micText.text = getString(R.string.mic_off)
    }

    private fun initOnnxModel() {
        try {
            val modelBytes = assets.open("cadence_model.onnx").readBytes()
            ortEnv = OrtEnvironment.getEnvironment()
            ortSession = ortEnv?.createSession(modelBytes)
            predictionText.text = getString(R.string.cadence_ml_ready)
        } catch (_: Exception) {
            predictionText.text = getString(R.string.cadence_ml_model_load_error)
        }
    }

    private fun cancelOrResetSafetyState() {
        mainHandler.removeCallbacks(graceCountdownRunnable)
        stopHelpSpeechListening()
        currentSafetyState = SafetyState.GREEN_MONITORING
        listeningStartMs = null
        graceStartMs = null
        silenceStartMs = null
        loudStreakCount = 0
        helpKeywordDetected = false
        alertSentForCurrentIncident = false
        statusDetailSuffix = ""
        recentSpikes.clear()
        lastTriggerReason = "cancelled_or_reset"
        applySafetyUi(SafetyState.GREEN_MONITORING, "")
        alertLogText.text = "Alert cancelled / state reset to normal monitoring."
    }

    private fun recordSpikeWindow(hasSpike: Boolean) {
        recentSpikes.addLast(hasSpike)
        if (recentSpikes.size > historySize) {
            recentSpikes.removeFirst()
        }
    }

    private fun enterYellowListeningState(reason: String) {
        if (currentSafetyState != SafetyState.GREEN_MONITORING) return
        currentSafetyState = SafetyState.YELLOW_LISTENING_HELP
        listeningStartMs = System.currentTimeMillis()
        helpKeywordDetected = false
        lastTriggerReason = "GREEN->YELLOW ($reason)"
        statusDetailSuffix = " — Listening for \"HELP\"!"
        triggerAlertVibration(force = false)
        applySafetyUi(currentSafetyState, statusDetailSuffix)

        val hasMic = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (hasMic) {
            if (!isMicMonitoring) {
                startMicMonitoring()
            }
            startHelpSpeechListening()
        }
    }

    private fun enterGracePeriod(reason: String) {
        if (currentSafetyState == SafetyState.RED_GRACE_PERIOD || currentSafetyState == SafetyState.ALERT_DISPATCHED) return
        stopHelpSpeechListening()
        currentSafetyState = SafetyState.RED_GRACE_PERIOD
        graceStartMs = System.currentTimeMillis()
        lastTriggerReason = "YELLOW->GRACE ($reason)"
        triggerAlertVibration(force = true)
        mainHandler.removeCallbacks(graceCountdownRunnable)
        mainHandler.post(graceCountdownRunnable)
    }

    private fun transitionToAlertDispatched(reason: String) {
        if (currentSafetyState == SafetyState.ALERT_DISPATCHED) return
        mainHandler.removeCallbacks(graceCountdownRunnable)
        stopHelpSpeechListening()
        currentSafetyState = SafetyState.ALERT_DISPATCHED
        statusDetailSuffix = ""
        lastTriggerReason = "GRACE->ALERT ($reason)"
        applySafetyUi(SafetyState.ALERT_DISPATCHED, "")
        dispatchEmergencyAlert()
    }

    private fun updateSafetyStateMachine(
        predictedLabel: String,
        jerkMax: Float,
        accelPeak: Float,
        variation: Float
    ) {
        lastPredictedLabel = predictedLabel
        lastJerkMax = jerkMax
        val now = System.currentTimeMillis()

        val isSuddenStopOrJerk = jerkMax >= jerkSpikeThreshold ||
                accelPeak >= suddenStopAccelJumpThreshold ||
                predictedLabel.equals("Sudden stop", ignoreCase = true)

        recordSpikeWindow(isSuddenStopOrJerk)
        val spikeCount = recentSpikes.count { it }

        when (currentSafetyState) {
            SafetyState.GREEN_MONITORING -> {
                if (spikeCount >= 2 || predictedLabel.equals("Sudden stop", ignoreCase = true)) {
                    enterYellowListeningState(
                        "sudden motion/stop (label=$predictedLabel, jerk=${String.format(Locale.US, "%.2f", jerkMax)})"
                    )
                }
            }

            SafetyState.YELLOW_LISTENING_HELP -> {
                val elapsed = now - (listeningStartMs ?: now)
                val remainingSec = ((maxHelpListenWindowMs - elapsed).coerceAtLeast(0L)) / 1000.0
                statusDetailSuffix = " — Listening for \"HELP\" (${String.format(Locale.US, "%.1f", remainingSec)}s)"
                applySafetyUi(currentSafetyState, statusDetailSuffix)

                if (helpKeywordDetected) {
                    enterGracePeriod("keyword 'help' matched")
                } else if (elapsed >= maxHelpListenWindowMs) {
                    stopHelpSpeechListening()
                    currentSafetyState = SafetyState.GREEN_MONITORING
                    listeningStartMs = null
                    statusDetailSuffix = ""
                    lastTriggerReason = "YELLOW->GREEN (no 'help' within ${maxHelpListenWindowMs / 1000}s)"
                    applySafetyUi(currentSafetyState, "")
                }
            }

            SafetyState.RED_GRACE_PERIOD -> {
                // Countdown handled by graceCountdownRunnable
            }

            SafetyState.ALERT_DISPATCHED -> {
                // Remains active until reset
            }
        }

        if (currentSafetyState == SafetyState.GREEN_MONITORING) {
            alertLogText.text = String.format(
                Locale.US,
                "jerkMax=%.2f | var=%.2f | %s",
                jerkMax,
                variation,
                lastTriggerReason
            )
        }
    }

    private fun applySafetyUi(state: SafetyState, suffix: String) {
        when (state) {
            SafetyState.GREEN_MONITORING -> {
                statusText.setTextColor("#2E7D32".toColorInt())
                statusText.text = "GREEN — Normal motion monitoring$suffix"
            }

            SafetyState.YELLOW_LISTENING_HELP -> {
                statusText.setTextColor("#EF6C00".toColorInt())
                statusText.text = "YELLOW — Sudden motion! Listening for \"HELP\"$suffix"
            }

            SafetyState.RED_GRACE_PERIOD -> {
                statusText.setTextColor("#C62828".toColorInt())
                statusText.text = "RED — \"HELP\" detected! Grace period$suffix"
            }

            SafetyState.ALERT_DISPATCHED -> {
                statusText.setTextColor("#D50000".toColorInt())
                statusText.text = "ALERT SENT — Local Police & Health dispatch payload ready"
            }
        }
    }

    private fun dispatchEmergencyAlert() {
        if (alertSentForCurrentIncident) return
        alertSentForCurrentIncident = true

        triggerAlertVibration(force = true)
        val timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US).format(Date())
        val payload = buildAlertPayload(timestamp)
        alertLogText.text = "$payload\nReason: $lastTriggerReason"
    }

    private fun buildAlertPayload(timestamp: String): String {
        val latStr = latestLat?.let { String.format(Locale.US, "%.6f", it) } ?: "unavailable"
        val lonStr = latestLon?.let { String.format(Locale.US, "%.6f", it) } ?: "unavailable"
        val speedKmh = latestSpeedMps * 3.6f
        val transcript = if (lastHelpHeardTranscript.isNotBlank()) lastHelpHeardTranscript else "help"

        return "EMERGENCY_DISPATCH | targets=LOCAL_POLICE,HEALTH_SERVICES | " +
                "label=$lastPredictedLabel | transcript=\"$transcript\" | " +
                "jerkMax=${String.format(Locale.US, "%.4f", lastJerkMax)} | " +
                "audioPeak=${peakAudioRms.toInt()} | lat=$latStr | lon=$lonStr | " +
                "speedKmh=${String.format(Locale.US, "%.1f", speedKmh)} | ts=$timestamp"
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        startLocationUpdates()
    }

    override fun onPause() {
        super.onPause()
        stopLocationUpdates()
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(graceCountdownRunnable)
        stopMicMonitoring()
        speechRecognizer?.destroy()
        speechRecognizer = null
        super.onDestroy()
        ortSession?.close()
        ortEnv?.close()
    }

    private fun calculateStdDev(values: Collection<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average().toFloat()
        val variance = values.asSequence().map { (it - mean).pow(2) }.average().toFloat()
        return sqrt(variance)
    }

    private fun buildFeatureVector17(): FloatArray {
        val accelList = accelerationWindow.toList()
        val accelAvg = accelList.average().toFloat()
        val accelPeak = accelList.maxOrNull() ?: 0f
        val accelMin = accelList.minOrNull() ?: 0f
        val accelRange = accelPeak - accelMin
        val accelBounce = calculateStdDev(accelList)

        val accelJerks = if (accelList.size > 1) {
            accelList.zipWithNext { a, b -> abs(b - a) }
        } else {
            listOf(0f)
        }
        val accelJerkMax = accelJerks.maxOrNull() ?: 0f
        val accelJerkAvg = accelJerks.average().toFloat()

        val halfIndex = accelList.size / 2
        val firstHalfAvg = if (halfIndex > 0) {
            accelList.subList(0, halfIndex).average().toFloat()
        } else {
            accelAvg
        }
        val secondHalfAvg = if (accelList.size - halfIndex > 0) {
            accelList.subList(halfIndex, accelList.size).average().toFloat()
        } else {
            accelAvg
        }
        val halfDrop = firstHalfAvg - secondHalfAvg

        val gyroList = gyroWindow.toList()
        val gyroAvg = if (gyroList.isNotEmpty()) gyroList.average().toFloat() else 0f
        val gyroPeak = gyroList.maxOrNull() ?: 0f
        val gyroMin = gyroList.minOrNull() ?: 0f
        val gyroRange = gyroPeak - gyroMin
        val gyroBounce = calculateStdDev(gyroList)

        val gyroJerks = if (gyroList.size > 1) {
            gyroList.zipWithNext { a, b -> abs(b - a) }
        } else {
            listOf(0f)
        }
        val gyroJerkMax = gyroJerks.maxOrNull() ?: 0f

        val accelGyroRatio = accelAvg / gyroAvg.coerceAtLeast(1e-4f)

        val xStd = calculateStdDev(xWindow)
        val yStd = calculateStdDev(yWindow)
        val zStd = calculateStdDev(zWindow)

        return floatArrayOf(
            accelAvg,
            accelBounce,
            accelPeak,
            accelMin,
            accelRange,
            accelJerkMax,
            accelJerkAvg,
            halfDrop,
            gyroAvg,
            gyroBounce,
            gyroPeak,
            gyroRange,
            gyroJerkMax,
            accelGyroRatio,
            xStd,
            yStd,
            zStd
        )
    }

    private fun runOnnxPrediction(features17: FloatArray, variation: Float) {
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
                    val predictedLabel = when (val rawOutput = results[0].value) {
                        is Array<*> -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        is LongArray -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        is FloatArray -> rawOutput.firstOrNull()?.toString() ?: "Unknown"
                        else -> rawOutput.toString()
                    }
                    val jerkMax = features17[5]
                    val accelPeak = features17[2]
                    val spikeFlag = if (jerkMax >= jerkSpikeThreshold) " [SPIKE]" else ""
                    predictionText.text = "Cadence ML: $predictedLabel$spikeFlag"
                    updateSafetyStateMachine(predictedLabel, jerkMax, accelPeak, variation)
                }
            }
        } catch (_: Exception) {
            predictionText.text = getString(R.string.cadence_ml_inference_error)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            val magnitude = sqrt(x * x + y * y + z * z)
            val dynamicAcceleration = abs(magnitude - 9.81f)

            accelerationWindow.addLast(dynamicAcceleration)
            xWindow.addLast(x)
            yWindow.addLast(y)
            zWindow.addLast(z)

            if (accelerationWindow.size > windowSize) accelerationWindow.removeFirst()
            if (xWindow.size > windowSize) xWindow.removeFirst()
            if (yWindow.size > windowSize) yWindow.removeFirst()
            if (zWindow.size > windowSize) zWindow.removeFirst()

            accelText.text = String.format(
                Locale.US,
                "X: %.2f\nY: %.2f\nZ: %.2f\nMagnitude: %.2f",
                x, y, z, magnitude
            )

            if (accelerationWindow.size < windowSize) {
                statusText.setTextColor(Color.WHITE)
                statusText.text = "Collecting motion data..."
            } else {
                val now = System.currentTimeMillis()
                if (now - lastEvalMs >= evalIntervalMs) {
                    lastEvalMs = now
                    val variation = calculateStdDev(accelerationWindow)
                    val features = buildFeatureVector17()
                    runOnnxPrediction(features, variation)
                }
            }
        }

        if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            latestGyroMagnitude = sqrt(x * x + y * y + z * z)
            gyroWindow.addLast(latestGyroMagnitude)
            if (gyroWindow.size > windowSize) gyroWindow.removeFirst()

            gyroText.text = String.format(
                Locale.US,
                "X: %.2f\nY: %.2f\nZ: %.2f\nRotation: %.2f",
                x, y, z, latestGyroMagnitude
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed
    }
}