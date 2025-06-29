import struct
import base64
from typing import List, Tuple
from phe import paillier

class PaillierKey:
    """Paillier encryption key with public and private key pair."""
    
    def __init__(self, public_key, private_key):
        if not hasattr(public_key, 'encrypt'):
            raise ValueError("Invalid public key")
        if not hasattr(private_key, 'decrypt'):
            raise ValueError("Invalid private key")
        
        self.public_key = public_key
        self.private_key = private_key
    
    @classmethod
    def generate_random(cls, n_length: int = 1024):
        """Generate a random Paillier key pair."""
        public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
        return cls(public_key, private_key)

class EncryptResult:
    """Result of Paillier vector encryption containing encrypted vector and normalization parameters."""
    
    def __init__(self, ciphertext: List[float], norm_params: str):
        self.ciphertext = ciphertext
        self.norm_params = norm_params

def _compress_norm_params(min_val: int, max_val: int) -> str:
    """Compress normalization parameters to a compact string."""
    min_bytes_lower = struct.pack('!Q', min_val & ((1 << 64) - 1))
    max_bytes_lower = struct.pack('!Q', max_val & ((1 << 64) - 1))
    min_bytes_upper = struct.pack('!Q', (min_val >> 64) & ((1 << 64) - 1))
    max_bytes_upper = struct.pack('!Q', (max_val >> 64) & ((1 << 64) - 1))
    all_bytes = min_bytes_lower + max_bytes_lower + min_bytes_upper + max_bytes_upper
    return base64.b64encode(all_bytes).decode('utf-8')

def _serialize_encrypted_vector(encrypted_vector: List) -> Tuple[List[float], str]:
    """Serialize encrypted vector for storage by normalizing to [-1, 1] range."""
    ciphertexts = [x.ciphertext() for x in encrypted_vector]
    
    if len(ciphertexts) > 0:
        max_val = max(ciphertexts)
        min_val = min(ciphertexts)
        range_val = max_val - min_val

        if range_val == 0:
            normalized = [0.0 for _ in ciphertexts]
        else:
            normalized = [2 * ((x - min_val) / range_val) - 1 for x in ciphertexts]
    else:
        normalized = []
        min_val = 0
        max_val = 0

    compressed_params = _compress_norm_params(min_val, max_val)
    return normalized, compressed_params

def _decompress_norm_params(compressed_str: str) -> Tuple[int, int]:
    """Decompress normalization parameters from string."""
    all_bytes = base64.b64decode(compressed_str)
    min_val_lower = struct.unpack('!Q', all_bytes[0:8])[0]
    max_val_lower = struct.unpack('!Q', all_bytes[8:16])[0]
    min_val_upper = struct.unpack('!Q', all_bytes[16:24])[0]
    max_val_upper = struct.unpack('!Q', all_bytes[24:32])[0]
    min_val = min_val_lower | (min_val_upper << 64)
    max_val = max_val_lower | (max_val_upper << 64)
    return min_val, max_val

def _deserialize_encrypted_vector(key: PaillierKey, normalized_vector: List[float], norm_params: str) -> List:
    """Convert normalized float vector back to Paillier encrypted numbers."""
    min_val, max_val = _decompress_norm_params(norm_params)

    if max_val != min_val:
        denormalized = [int(((x + 1) / 2) * (max_val - min_val) + min_val) for x in normalized_vector]
    else:
        denormalized = [min_val for _ in normalized_vector]

    recreated_vector = [paillier.EncryptedNumber(key.public_key, x) for x in denormalized]
    return recreated_vector

def encrypt_vector(key: PaillierKey, message: List[float], scaling_factor: int = 1000) -> EncryptResult:
    """Encrypt a vector using Paillier homomorphic encryption."""
    if not isinstance(message, list):
        raise ValueError("Message must be a list")
    if not all(isinstance(x, (int, float)) for x in message):
        raise ValueError("All message elements must be numbers")
    if len(message) == 0:
        return EncryptResult([], _compress_norm_params(0, 0))

    scaled_embedding = [int(round(x * scaling_factor)) for x in message]
    encrypted_vector = [key.public_key.encrypt(x) for x in scaled_embedding]
    normalized_vector, compressed_params = _serialize_encrypted_vector(encrypted_vector)
    return EncryptResult(normalized_vector, compressed_params)

def homomorphic_add_vectors(key: PaillierKey, encrypted_result1: EncryptResult, encrypted_result2: EncryptResult) -> EncryptResult:
    """Perform homomorphic addition of two encrypted vectors."""
    if len(encrypted_result1.ciphertext) != len(encrypted_result2.ciphertext):
        raise ValueError("Vectors must have the same dimension")

    if len(encrypted_result1.ciphertext) == 0:
        return EncryptResult([], _compress_norm_params(0, 0))

    enc_vec1 = _deserialize_encrypted_vector(key, encrypted_result1.ciphertext, encrypted_result1.norm_params)
    enc_vec2 = _deserialize_encrypted_vector(key, encrypted_result2.ciphertext, encrypted_result2.norm_params)

    result_encrypted = [a + b for a, b in zip(enc_vec1, enc_vec2)]
    normalized_result, compressed_params = _serialize_encrypted_vector(result_encrypted)
    return EncryptResult(normalized_result, compressed_params)

def homomorphic_scalar_multiply(key: PaillierKey, encrypted_result: EncryptResult, scalar: int) -> EncryptResult:
    """Perform homomorphic scalar multiplication of an encrypted vector."""
    if len(encrypted_result.ciphertext) == 0:
        return EncryptResult([], _compress_norm_params(0, 0))

    encrypted_vector = _deserialize_encrypted_vector(key, encrypted_result.ciphertext, encrypted_result.norm_params)
    result_encrypted = [scalar * x for x in encrypted_vector]
    normalized_result, compressed_params = _serialize_encrypted_vector(result_encrypted)
    return EncryptResult(normalized_result, compressed_params)

# Convenience functions for easy testing
def generate_random_key(n_length: int = 1024) -> PaillierKey:
    """Generate a random Paillier key for testing."""
    return PaillierKey.generate_random(n_length)

def test_paillier_encryption():
    """Simple test function to verify Paillier encryption and homomorphic operations."""
    key = generate_random_key(n_length=1024)
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    scaling_factor = 1000

    encrypted_result = encrypt_vector(key, test_vector, scaling_factor)
    print(f"Original vector: {test_vector}")
    print(f"Encrypted vector (normalized): {encrypted_result.ciphertext}")

    test_homomorphic_operations(key, scaling_factor)

def test_homomorphic_operations(key: PaillierKey, scaling_factor: int = 1000):
    """Test homomorphic addition and scalar multiplication."""
    print("\nTesting Homomorphic Operations:")
    print("-" * 40)

    vector1 = [1.0, 2.0, 3.0]
    vector2 = [0.5, 1.5, 2.5]
    scalar = 3

    enc1 = encrypt_vector(key, vector1, scaling_factor)
    enc2 = encrypt_vector(key, vector2, scaling_factor)

    enc_sum = homomorphic_add_vectors(key, enc1, enc2)
    print(f"Encrypted sum (normalized): {enc_sum.ciphertext}")

    enc_scaled = homomorphic_scalar_multiply(key, enc1, scalar)
    print(f"Encrypted scalar multiplication (normalized): {enc_scaled.ciphertext}")

if __name__ == "__main__":
    test_paillier_encryption()
