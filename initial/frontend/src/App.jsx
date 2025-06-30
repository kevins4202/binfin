import React, { useState, useCallback } from "react";
import {
  Upload,
  FileAudio,
  AlertCircle,
  CheckCircle,
  XCircle,
  Loader2,
  Play,
  Pause,
} from "lucide-react";
import "./App.css";

const AudioDeepfakeDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Configuration - update this for production
  const API_BASE_URL = "http://localhost:8000";

  const handleFileSelect = useCallback((file) => {
    if (!file.name.toLowerCase().endsWith(".wav")) {
      setError("Please select a .wav file");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      // 50MB limit
      setError("File size must be less than 50MB");
      return;
    }

    setSelectedFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect]
  );

  const handleFileInputChange = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (file) {
        handleFileSelect(file);
      }
    },
    [handleFileSelect]
  );

  const analyzeAudio = async () => {
    if (!selectedFile) {
      setError("Please select a file first");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Analysis failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setError(error.message || "Failed to analyze audio. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const toggleAudio = () => {
    const audio = document.getElementById("audioElement");
    if (audio) {
      if (isPlaying) {
        audio.pause();
      } else {
        audio.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="app-container">
      <div className="main-card">
        {/* Header */}
        <div className="header">
          <div className="header-content">
            <FileAudio className="header-icon" />
            <h1 className="header-title">
              Audio Deepfake Detector
            </h1>
          </div>
          <p className="header-subtitle">
            Upload a .wav file to detect if it's real or AI-generated
          </p>
        </div>

        {/* Upload Area */}
        <div
          className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <Upload className="upload-icon" />
          <p className="upload-title">
            Click to upload or drag & drop
          </p>
          <p className="upload-subtitle">
            Only .wav files are supported (max 50MB)
          </p>
        </div>

        <input
          type="file"
          id="fileInput"
          accept=".wav"
          onChange={handleFileInputChange}
          className="hidden-input"
        />

        {/* File Info */}
        {selectedFile && (
          <div className="file-info">
            <div className="file-info-content">
              <div className="file-details">
                <h3>{selectedFile.name}</h3>
                <p>{formatFileSize(selectedFile.size)}</p>
              </div>
              {audioUrl && (
                <button
                  onClick={toggleAudio}
                  className="play-button"
                >
                  {isPlaying ? (
                    <Pause />
                  ) : (
                    <Play />
                  )}
                  {isPlaying ? "Pause" : "Play"}
                </button>
              )}
            </div>

            {audioUrl && (
              <audio
                id="audioElement"
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                onPause={() => setIsPlaying(false)}
                onPlay={() => setIsPlaying(true)}
                className="hidden-audio"
              />
            )}
          </div>
        )}

        {/* Analyze Button */}
        <div className="analyze-section">
          <button
            onClick={analyzeAudio}
            disabled={!selectedFile || isAnalyzing}
            className="analyze-button"
          >
            {isAnalyzing ? (
              <div className="analyze-button-content">
                <Loader2 className="spinner" />
                Analyzing...
              </div>
            ) : (
              "Analyze Audio"
            )}
          </button>
        </div>

        {/* Loading State */}
        {isAnalyzing && (
          <div className="loading-state">
            <div className="loading-content">
              <Loader2 className="loading-spinner" />
              Analyzing audio... This may take a moment.
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className={`results ${result.prediction === "REAL" ? "real" : "fake"}`}>
            <div className="results-header">
              {result.prediction === "REAL" ? (
                <CheckCircle className="results-icon" />
              ) : (
                <XCircle className="results-icon" />
              )}
              <h3 className="results-title">
                Prediction: {result.prediction}
              </h3>
            </div>

            <p className="results-confidence">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>

            <div className="results-grid">
              <div className="result-item">
                <p className="result-label">Real Audio</p>
                <p className="result-value">
                  {(result.real_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="result-item">
                <p className="result-label">Fake Audio</p>
                <p className="result-value">
                  {(result.fake_probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <div className="error-content">
              <AlertCircle className="error-icon" />
              <p>{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioDeepfakeDetector;
