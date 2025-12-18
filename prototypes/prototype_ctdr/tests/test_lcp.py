"""
Tests for DPX LCP Kernel
Compares CUDA implementation with CPU baseline
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import lcp_cpu
from src.encoding import encode_to_short2, decode_from_short2
from cuda.kernels import run_dpx_lcp_with_fallback


class TestEncoding:
    """Tests for short2 encoding/decoding"""
    
    def test_encode_decode_simple(self):
        """Test basic encode/decode roundtrip"""
        text = "hello"
        encoded = encode_to_short2(text)
        decoded = decode_from_short2(encoded)
        assert decoded == text
    
    def test_encode_decode_empty(self):
        """Test empty string"""
        text = ""
        encoded = encode_to_short2(text)
        decoded = decode_from_short2(encoded)
        assert decoded == text
    
    def test_encode_decode_odd_length(self):
        """Test odd-length string (padding)"""
        text = "abc"
        encoded = encode_to_short2(text)
        decoded = decode_from_short2(encoded)
        assert decoded == text
    
    def test_encode_decode_long(self):
        """Test long string"""
        text = "a" * 1000
        encoded = encode_to_short2(text)
        decoded = decode_from_short2(encoded)
        assert decoded == text


class TestLCP:
    """Tests for LCP computation (CUDA vs CPU)"""
    
    def test_lcp_hello_hell(self):
        """Test: LCP('hello', 'hell') = 4"""
        s1 = "hello"
        s2 = "hell"
        expected = 4
        
        # CPU baseline
        cpu_result = lcp_cpu(s1, s2)
        assert cpu_result == expected, f"CPU: expected {expected}, got {cpu_result}"
        
        # CUDA (with fallback)
        s1_encoded = encode_to_short2(s1)
        s2_encoded = encode_to_short2(s2)
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == expected, f"CUDA: expected {expected}, got {cuda_result}"
    
    def test_lcp_world_word(self):
        """Test: LCP('world', 'word') = 3"""
        s1 = "world"
        s2 = "word"
        expected = 3
        
        cpu_result = lcp_cpu(s1, s2)
        assert cpu_result == expected
        
        s1_encoded = encode_to_short2(s1)
        s2_encoded = encode_to_short2(s2)
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == expected
    
    def test_lcp_test_test(self):
        """Test: LCP('test', 'test') = 4"""
        s1 = "test"
        s2 = "test"
        expected = 4
        
        cpu_result = lcp_cpu(s1, s2)
        assert cpu_result == expected
        
        s1_encoded = encode_to_short2(s1)
        s2_encoded = encode_to_short2(s2)
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == expected
    
    def test_lcp_abc_xyz(self):
        """Test: LCP('abc', 'xyz') = 0"""
        s1 = "abc"
        s2 = "xyz"
        expected = 0
        
        cpu_result = lcp_cpu(s1, s2)
        assert cpu_result == expected
        
        s1_encoded = encode_to_short2(s1)
        s2_encoded = encode_to_short2(s2)
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == expected
    
    def test_lcp_empty_strings(self):
        """Test: LCP('', '') = 0, LCP('', 'abc') = 0"""
        # Both empty
        cpu_result = lcp_cpu("", "")
        assert cpu_result == 0
        
        s1_encoded = encode_to_short2("")
        s2_encoded = encode_to_short2("")
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == 0
        
        # One empty
        cpu_result = lcp_cpu("", "abc")
        assert cpu_result == 0
        
        s1_encoded = encode_to_short2("")
        s2_encoded = encode_to_short2("abc")
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == 0
    
    def test_lcp_long_strings(self):
        """Test: LCP with very long strings (1000+ chars)"""
        s1 = "a" * 1000 + "b"
        s2 = "a" * 1000 + "c"
        expected = 1000
        
        cpu_result = lcp_cpu(s1, s2)
        assert cpu_result == expected
        
        s1_encoded = encode_to_short2(s1)
        s2_encoded = encode_to_short2(s2)
        cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
        assert cuda_result == expected
    
    def test_lcp_cuda_vs_cpu_consistency(self):
        """Test: CUDA results match CPU baseline for multiple cases"""
        test_cases = [
            ("hello", "hell", 4),
            ("world", "word", 3),
            ("test", "test", 4),
            ("abc", "xyz", 0),
            ("", "", 0),
            ("", "abc", 0),
            ("prefix", "pre", 3),
            ("a", "a", 1),
            ("a", "b", 0),
        ]
        
        for s1, s2, expected in test_cases:
            cpu_result = lcp_cpu(s1, s2)
            assert cpu_result == expected, f"CPU failed for ('{s1}', '{s2}')"
            
            s1_encoded = encode_to_short2(s1)
            s2_encoded = encode_to_short2(s2)
            cuda_result = run_dpx_lcp_with_fallback(s1_encoded, s2_encoded)
            
            assert cuda_result == expected, (
                f"CUDA failed for ('{s1}', '{s2}'): "
                f"expected {expected}, got {cuda_result}"
            )
            
            # Also verify CUDA matches CPU
            assert cuda_result == cpu_result, (
                f"Mismatch for ('{s1}', '{s2}'): "
                f"CPU={cpu_result}, CUDA={cuda_result}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
