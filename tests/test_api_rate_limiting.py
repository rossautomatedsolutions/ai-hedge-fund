import os
import pytest
from unittest.mock import Mock, patch, call

from src.tools.api import _make_api_request, get_prices
import requests
from src.data.cache import Cache

class TestRateLimiting:
    """Test suite for API rate limiting functionality."""

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_handles_single_rate_limit(self, mock_get, mock_sleep):
        """Test that API retries once after a 429 and succeeds."""
        # Setup mock responses: first 429, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"
        
        mock_get.side_effect = [mock_429_response, mock_200_response]
        
        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        
        result = _make_api_request(url, headers, max_retries=3)
        
        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"
        
        # Verify requests.get was called twice
        assert mock_get.call_count == 2
        mock_get.assert_has_calls([
            call(url, headers=headers, timeout=15),
            call(url, headers=headers, timeout=15)
        ])
        
        # Verify sleep was called once with 1 second (first retry)
        mock_sleep.assert_called_once_with(1.0)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_handles_multiple_rate_limits(self, mock_get, mock_sleep):
        """Test that API retries multiple times after 429s."""
        # Setup mock responses: three 429s, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"
        
        mock_get.side_effect = [
            mock_429_response, 
            mock_429_response, 
            mock_429_response, 
            mock_200_response
        ]
        
        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        
        result = _make_api_request(url, headers, max_retries=3)
        
        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"
        
        # Verify requests.get was called 4 times
        assert mock_get.call_count == 4
        
        # Verify sleep was called 3 times with exponential backoff: 1s, 2s, 4s
        assert mock_sleep.call_count == 3
        expected_calls = [call(1.0), call(2.0), call(4.0)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.post')
    def test_handles_post_rate_limiting(self, mock_post, mock_sleep):
        """Test that POST requests handle rate limiting."""
        # Setup mock responses: first 429, then 200
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"
        
        mock_post.side_effect = [mock_429_response, mock_200_response]
        
        # Call the function with POST method
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        json_data = {"test": "data"}
        
        result = _make_api_request(url, headers, method="POST", json_data=json_data)
        
        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"
        
        # Verify requests.post was called twice
        assert mock_post.call_count == 2
        mock_post.assert_has_calls([
            call(url, headers=headers, json=json_data, timeout=15),
            call(url, headers=headers, json=json_data, timeout=15)
        ])
        
        # Verify sleep was called once with 1 second (first retry)
        mock_sleep.assert_called_once_with(1.0)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_ignores_other_errors(self, mock_get, mock_sleep):
        """Test that non-429 errors are returned without retrying."""
        # Setup mock response: 500 error followed by success
        mock_500_response = Mock()
        mock_500_response.status_code = 500
        mock_500_response.text = "Internal Server Error"

        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"
        
        mock_get.side_effect = [mock_500_response, mock_200_response]
        
        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        
        result = _make_api_request(url, headers)
        
        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"
        
        # Verify requests.get was called twice
        assert mock_get.call_count == 2
        
        # Verify sleep was called with the first backoff interval
        mock_sleep.assert_called_once_with(1.0)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_normal_success_requests(self, mock_get, mock_sleep):
        """Test that successful requests return immediately without retry."""
        # Setup mock response: 200 success
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"
        
        mock_get.return_value = mock_200_response
        
        # Call the function
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        
        result = _make_api_request(url, headers)
        
        # Verify behavior
        assert result.status_code == 200
        assert result.text == "Success"
        
        # Verify requests.get was called only once
        assert mock_get.call_count == 1
        
        # Verify sleep was never called
        mock_sleep.assert_not_called()

    @patch('src.tools.api._cache')
    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_full_integration(self, mock_get, mock_sleep, mock_cache):
        """Test that get_prices function properly handles rate limiting."""
        # Mock cache to return None (cache miss)
        mock_cache.get_prices.return_value = None
        
        # Setup mock responses: first 429, then 200 with valid data
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.json.return_value = {
            "ticker": "AAPL",
            "prices": [
                {
                    "time": "2024-01-01T00:00:00Z",
                    "open": 100.0,
                    "close": 101.0,
                    "high": 102.0,
                    "low": 99.0,
                    "volume": 1000
                }
            ]
        }
        
        mock_get.side_effect = [mock_429_response, mock_200_response]
        
        # Set environment variable for API key
        with patch.dict(os.environ, {"FINANCIAL_DATASETS_API_KEY": "test-key"}):
            # Call get_prices
            result = get_prices("AAPL", "2024-01-01", "2024-01-02")
        
        # Verify the function succeeded and returned data
        assert len(result) == 1
        assert result[0].open == 100.0
        assert result[0].close == 101.0
        
        # Verify rate limiting behavior
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)
        
        # Verify cache operations
        mock_cache.get_prices.assert_called_once()
        mock_cache.set_prices.assert_called_once()

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test that function stops retrying after max_retries and returns final 429."""
        # Setup mock responses: all 429s (exceeds max retries)
        mock_429_response = Mock()
        mock_429_response.status_code = 429
        mock_429_response.text = "Too Many Requests"
        
        mock_get.return_value = mock_429_response
        
        # Call the function with max_retries=2
        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/test"
        
        result = _make_api_request(url, headers, max_retries=2)
        
        # Verify final 429 is returned
        assert result.status_code == 429
        assert result.text == "Too Many Requests"
        
        # Verify requests.get was called 3 times (1 initial + 2 retries)
        assert mock_get.call_count == 3
        
        # Verify sleep was called 2 times with exponential backoff: 1s, 2s
        assert mock_sleep.call_count == 2
        expected_calls = [call(1.0), call(2.0)]
        mock_sleep.assert_has_calls(expected_calls)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_retries_ssl_error_then_succeeds(self, mock_get, mock_sleep):
        """Transient SSL failures should be retried."""
        mock_200_response = Mock()
        mock_200_response.status_code = 200
        mock_200_response.text = "Success"

        mock_get.side_effect = [
            requests.exceptions.SSLError("EOF occurred in violation of protocol"),
            mock_200_response,
        ]

        result = _make_api_request("https://api.financialdatasets.ai/test", {"X-API-KEY": "test-key"})

        assert result.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @patch('src.tools.api.time.sleep')
    @patch('src.tools.api.requests.get')
    def test_does_not_retry_401_missing_key(self, mock_get, mock_sleep):
        """Deterministic auth failures should not be retried."""
        mock_401_response = Mock()
        mock_401_response.status_code = 401
        mock_401_response.text = '{"error":"Missing API key"}'

        mock_get.return_value = mock_401_response

        result = _make_api_request("https://api.financialdatasets.ai/test", {})

        assert result.status_code == 401
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    def test_identical_get_is_served_from_request_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("src.tools.api._cache", Cache())

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prices": [{"time": "2024-01-01T00:00:00Z"}]}
        mock_response.text = '{"prices":[{"time":"2024-01-01T00:00:00Z"}]}'

        mock_get = Mock(return_value=mock_response)
        monkeypatch.setattr("src.tools.api.requests.get", mock_get)

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/prices/?ticker=AAPL"

        first = _make_api_request(url, headers)
        second = _make_api_request(url, headers)

        assert first.json() == second.json()
        assert mock_get.call_count == 1

    def test_identical_post_is_served_from_request_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("src.tools.api._cache", Cache())

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"search_results": [{"ticker": "AAPL", "report_period": "2024-12-31", "period": "ttm", "currency": "USD"}]}
        mock_response.text = '{"search_results":[{"ticker":"AAPL","report_period":"2024-12-31","period":"ttm","currency":"USD"}]}'

        mock_post = Mock(return_value=mock_response)
        monkeypatch.setattr("src.tools.api.requests.post", mock_post)

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/financials/search/line-items"
        body = {"tickers": ["AAPL"], "line_items": ["revenue"], "end_date": "2025-01-01", "period": "ttm", "limit": 5}

        first = _make_api_request(url, headers, method="POST", json_data=body)
        second = _make_api_request(url, headers, method="POST", json_data=body)

        assert first.json() == second.json()
        assert mock_post.call_count == 1

    def test_transient_failure_is_not_cached(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("src.tools.api._cache", Cache())

        mock_get = Mock(side_effect=requests.exceptions.ConnectionError("ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')"))
        monkeypatch.setattr("src.tools.api.requests.get", mock_get)

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/prices/?ticker=AAPL"

        with pytest.raises(requests.exceptions.ConnectionError):
            _make_api_request(url, headers, max_retries=0)

        with pytest.raises(requests.exceptions.ConnectionError):
            _make_api_request(url, headers, max_retries=0)

        assert mock_get.call_count == 2

    def test_different_post_bodies_use_different_cache_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("src.tools.api._cache", Cache())

        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = {"search_results": [{"ticker": "AAPL", "report_period": "2024-12-31", "period": "ttm", "currency": "USD", "revenue": 100}]}
        first_response.text = '{"search_results":[{"ticker":"AAPL","report_period":"2024-12-31","period":"ttm","currency":"USD","revenue":100}]}'

        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = {"search_results": [{"ticker": "AAPL", "report_period": "2024-12-31", "period": "ttm", "currency": "USD", "net_income": 50}]}
        second_response.text = '{"search_results":[{"ticker":"AAPL","report_period":"2024-12-31","period":"ttm","currency":"USD","net_income":50}]}'

        mock_post = Mock(side_effect=[first_response, second_response])
        monkeypatch.setattr("src.tools.api.requests.post", mock_post)

        headers = {"X-API-KEY": "test-key"}
        url = "https://api.financialdatasets.ai/financials/search/line-items"
        first_body = {"tickers": ["AAPL"], "line_items": ["revenue"], "end_date": "2025-01-01", "period": "ttm", "limit": 5}
        second_body = {"tickers": ["AAPL"], "line_items": ["net_income"], "end_date": "2025-01-01", "period": "ttm", "limit": 5}

        first = _make_api_request(url, headers, method="POST", json_data=first_body)
        second = _make_api_request(url, headers, method="POST", json_data=second_body)

        assert first.json() != second.json()
        assert mock_post.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__]) 
