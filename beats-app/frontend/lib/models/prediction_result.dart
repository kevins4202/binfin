class PredictionResult {
  final String prediction;
  final double confidence;
  final Map<String, double> probabilities;

  PredictionResult({
    required this.prediction,
    required this.confidence,
    required this.probabilities,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      prediction: json['prediction'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      probabilities: Map<String, double>.from(json['probabilities']),
    );
  }
} 