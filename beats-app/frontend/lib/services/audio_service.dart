import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AudioService {
  static const List<String> baseUrls = [
    'http://127.0.0.1:5015',
    'http://localhost:5015',
    'http://0.0.0.0:5015',
  ];
  
  static Future<Map<String, dynamic>> uploadAudioFile(File file) async {
    Exception? lastException;
    
    for (String baseUrl in baseUrls) {
      try {
        print('Trying to connect to: $baseUrl');
        
        // Create multipart request
        var request = http.MultipartRequest(
          'POST',
          Uri.parse('$baseUrl/api/predict'),
        );
        
        // Add file to request
        request.files.add(
          await http.MultipartFile.fromPath(
            'file',
            file.path,
          ),
        );
        
        // Send request with timeout
        var response = await request.send().timeout(
          const Duration(seconds: 30),
          onTimeout: () {
            throw Exception('Request timeout');
          },
        );
        
        var responseBody = await response.stream.bytesToString();
        
        if (response.statusCode == 200) {
          print('Successfully connected to: $baseUrl');
          return json.decode(responseBody);
        } else {
          throw Exception('Failed to upload file: ${response.statusCode} - $responseBody');
        }
      } catch (e) {
        print('Failed to connect to $baseUrl: $e');
        lastException = Exception('Failed to connect to $baseUrl: $e');
        continue;
      }
    }
    
    throw lastException ?? Exception('Failed to connect to any server');
  }
} 