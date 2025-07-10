import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'services/audio_service.dart';
import 'models/prediction_result.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio Deepfake Detection App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const AudioUploadScreen(),
    );
  }
}

class AudioUploadScreen extends StatefulWidget {
  const AudioUploadScreen({super.key});

  @override
  State<AudioUploadScreen> createState() => _AudioUploadScreenState();
}

class _AudioUploadScreenState extends State<AudioUploadScreen> {
  Uint8List? selectedFileBytes;
  String? selectedFileName;
  PredictionResult? predictionResult;
  bool isLoading = false;
  String? errorMessage;

  Future<void> pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
      );

      if (result != null) {
        setState(() {
          selectedFileBytes = result.files.single.bytes;
          selectedFileName = result.files.single.name;
          predictionResult = null;
          errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Error picking file: $e';
      });
    }
  }

  Future<void> uploadAndPredict() async {
    if (selectedFileBytes == null) {
      setState(() {
        errorMessage = 'Please select a WAV file first';
      });
      return;
    }

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final response = await AudioService.uploadAudioFile(
        selectedFileBytes!,
        selectedFileName,
      );
      final result = PredictionResult.fromJson(response);
      
      setState(() {
        predictionResult = result;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = 'Error: $e';
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Prediction'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // File selection section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Select WAV File',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: pickFile,
                            icon: const Icon(Icons.upload_file),
                            label: const Text('Choose WAV File'),
                          ),
                        ),
                        const SizedBox(width: 16),
                        if (selectedFileBytes != null)
                          Expanded(
                            child: Text(
                              selectedFileName ?? 'Unknown file',
                              style: const TextStyle(
                                fontSize: 14,
                                color: Colors.grey,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Upload and predict button
            if (selectedFileBytes != null)
              ElevatedButton(
                onPressed: isLoading ? null : uploadAndPredict,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: isLoading
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                          SizedBox(width: 16),
                          Text('Analyzing...'),
                        ],
                      )
                    : const Text('Analyze Audio'),
              ),
            
            const SizedBox(height: 16),
            
            // Error message
            if (errorMessage != null)
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Icon(Icons.error, color: Colors.red.shade700),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          errorMessage!,
                          style: TextStyle(color: Colors.red.shade700),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            
            const SizedBox(height: 16),
            
            // Prediction result
            if (predictionResult != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Prediction Result',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      
                      // Prediction and confidence
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: predictionResult!.prediction == 'real'
                                  ? Colors.green.shade100
                                  : Colors.red.shade100,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Text(
                              predictionResult!.prediction.toUpperCase(),
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: predictionResult!.prediction == 'real'
                                    ? Colors.green.shade800
                                    : Colors.red.shade800,
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Text(
                              'Confidence: ${(predictionResult!.confidence * 100).toStringAsFixed(1)}%',
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ],
                      ),
                      
                      const SizedBox(height: 16),
                      
                      // Probability bars
                      const Text(
                        'Probabilities',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 8),
                      
                      // Real probability
                      Row(
                        children: [
                          const Text('Real:', style: TextStyle(fontSize: 14)),
                          const SizedBox(width: 8),
                          Expanded(
                            child: LinearProgressIndicator(
                              value: predictionResult!.probabilities['real'] ?? 0,
                              backgroundColor: Colors.grey.shade200,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                Colors.green.shade400,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${((predictionResult!.probabilities['real'] ?? 0) * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                      
                      const SizedBox(height: 8),
                      
                      // Fake probability
                      Row(
                        children: [
                          const Text('Fake:', style: TextStyle(fontSize: 14)),
                          const SizedBox(width: 8),
                          Expanded(
                            child: LinearProgressIndicator(
                              value: predictionResult!.probabilities['fake'] ?? 0,
                              backgroundColor: Colors.grey.shade200,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                Colors.red.shade400,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${((predictionResult!.probabilities['fake'] ?? 0) * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
