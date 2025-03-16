import unittest
import json
from unittest.mock import patch, mock_open
import requests

from api_call import load_config, make_api_call  # Import functions from notebook code

class TestAPI(unittest.TestCase):

    @patch("../config/api-call.json", new_callable=mock_open, read_data='{"ned_api_endpoint": "https://example.com", "API_KEY": "test123"}')
    def test_load_config(self, mock_file):
        """Test if config loads correctly from JSON"""
        config = load_config()
        self.assertEqual(config["ned"]["ned_api_endpoint"], "https://api.ned.nl/v1/utilizations")
        self.assertEqual(config["ned"]["demo-ned-api-key"], "test123")

    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open, read_data='{"API_URL": "https://example.com", "API_KEY": "test123"}')
    def test_make_api_call(self, mock_file, mock_requests):
        """Test API call with mocked response"""
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Success"}
        mock_requests.return_value = mock_response

        response = make_api_call()
        self.assertEqual(response, {"message": "Success"})

    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open, read_data='{"API_URL": "https://example.com", "API_KEY": "test123"}')
    def test_make_api_call_failure(self, mock_file, mock_requests):
        """Test API call failure (non-200 response)"""
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 404
        mock_requests.return_value = mock_response

        response = make_api_call()
        self.assertIsNone(response)

if __name__ == "__main__":
    unittest.main()
