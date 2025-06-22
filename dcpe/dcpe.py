import numpy as np
import secrets
import hashlib
import hmac
import struct
from typing import List
from Crypto.Cipher import ChaCha20

class DCPEKey:
    """Simplified DCPE encryption key combining encryption key bytes and scaling factor."""
    
    def __init__(self, key_bytes: bytes, scaling_factor: float):
        if not isinstance(key_bytes, bytes):
            raise ValueError("Key must be bytes")
        if len(key_bytes) != 32:
            raise ValueError("Key must be exactly 32 bytes (for ChaCha20 and HMAC-SHA256)")
        if not isinstance(scaling_factor, float):
            raise ValueError("Scaling factor must be float")
        if scaling_factor == 0.0:
            raise ValueError("Scaling factor cannot be zero")
        
        self.key_bytes = key_bytes
        self.scaling_factor = scaling_factor
    
    @classmethod
    def generate_random(cls, scaling_factor: float = 1.0):
        """Generate a random DCPE key with specified scaling factor."""
        return cls(secrets.token_bytes(32), scaling_factor)

class AuthHash:
    """Simple authentication hash for vector integrity."""
    
    def __init__(self, hash_bytes: bytes):
        if len(hash_bytes) != 32:  # HMAC-SHA-256 hash
            raise ValueError("Auth hash must be 32 bytes (HMAC-SHA-256)")
        self.hash_bytes = hash_bytes
    
    def get_bytes(self) -> bytes:
        return self.hash_bytes

class EncryptResult:
    """Result of DCPE vector encryption containing ciphertext, IV, and auth hash."""
    
    def __init__(self, ciphertext: List[float], iv: bytes, auth_hash: AuthHash):
        self.ciphertext = ciphertext
        self.iv = iv
        self.auth_hash = auth_hash


def shuffle(key: DCPEKey, input_list: List[float]) -> List[float]:
    """Deterministically shuffle a list based on the encryption key."""
    if not input_list:
        return input_list
    
    key_material = key.key_bytes
    indexed_input = list(enumerate(input_list))
    
    def get_sort_key(index_value_pair):
        index, _ = index_value_pair
        # Use HMAC for a keyed hash to generate sort keys.
        # The message is just the index, the key is the main secret key.
        sort_key_mac = hmac.new(key_material, index.to_bytes(4, 'big'), hashlib.sha256)
        return int.from_bytes(sort_key_mac.digest()[:8], 'big')
    
    shuffled_indexed = sorted(indexed_input, key=get_sort_key)
    return [value for _, value in shuffled_indexed]


def unshuffle(key: DCPEKey, input_list: List[float]) -> List[float]:
    """Reverse the deterministic shuffle operation."""
    if not input_list:
        return input_list
    
    key_material = key.key_bytes
    indexed_input = list(enumerate(input_list))
    
    def get_sort_key(index_value_pair):
        index, _ = index_value_pair
        sort_key_mac = hmac.new(key_material, index.to_bytes(4, 'big'), hashlib.sha256)
        return int.from_bytes(sort_key_mac.digest()[:8], 'big')
        
    shuffled_indexed = sorted(indexed_input, key=get_sort_key)
    
    unshuffled_list = [0.0] * len(input_list)
    for shuffled_pos, (original_pos, value) in enumerate(shuffled_indexed):
        unshuffled_list[original_pos] = input_list[shuffled_pos]
    
    return unshuffled_list


def _generate_noise_vector(key: DCPEKey, iv: bytes, approximation_factor: float, dimensionality: int) -> np.ndarray:
    """
    Generate a deterministic noise vector using a standard stream cipher (ChaCha20).
    """
    # We need enough random data for the normal vector components and one uniform point.
    # Each value will be derived from an 8-byte (64-bit) integer.
    num_random_values = dimensionality + 1
    bytes_needed = num_random_values * 8

    # Use ChaCha20 as a standard, secure pseudorandom generator (PRG)
    cipher = ChaCha20.new(key=key.key_bytes, nonce=iv)
    keystream = cipher.encrypt(bytes(bytes_needed)) # Encrypting zeros gives the raw keystream

    # Unpack the raw bytes into unsigned 64-bit integers. Use big-endian '>' for consistency.
    random_ints = struct.unpack(f'>{num_random_values}Q', keystream)
    
    # Convert the large integers into uniform floats in the range [0.0, 1.0)
    max_int = 2**64
    uniform_floats = [val / max_int for val in random_ints]
    
    # Use the securely generated uniform floats to create the noise vector components
    normal_vector_components = uniform_floats[:dimensionality]
    uniform_point_for_magnitude = uniform_floats[dimensionality]
    
    normal_vector = np.zeros(dimensionality)
    for i in range(dimensionality):
        u = normal_vector_components[i]
        # Box-Muller-like transform to generate normally distributed values
        normal_vector[i] = np.sqrt(-2 * np.log(max(u, 1e-10))) * np.cos(2 * np.pi * u)
    
    # Calculate noise parameters
    d_dimensional_ball_radius = key.scaling_factor / 4.0 * approximation_factor
    uniform_point_in_ball = d_dimensional_ball_radius * (uniform_point_for_magnitude ** (1.0 / dimensionality))
    
    # Normalize and scale
    norm = np.linalg.norm(normal_vector)
    if norm > 0:
        normalized_vector = normal_vector * uniform_point_in_ball / norm
    else:
        normalized_vector = np.zeros(dimensionality)
    
    return normalized_vector

def _compute_auth_hash(key: DCPEKey, approximation_factor: float, iv: bytes, encrypted_embedding: List[float]) -> AuthHash:
    """
    Compute authentication hash using the standard HMAC-SHA256.
    """
    # The key for HMAC is the secret key. The message is all the non-secret data.
    mac = hmac.new(key.key_bytes, digestmod=hashlib.sha256)
    
    # Update the MAC with all data that needs to be authenticated.
    # Use struct.pack for a canonical, fixed-size representation of floats.
    mac.update(struct.pack('d', key.scaling_factor))
    mac.update(struct.pack('d', approximation_factor))
    mac.update(iv)
    
    for value in encrypted_embedding:
        mac.update(struct.pack('d', value))
    
    return AuthHash(mac.digest())

def _verify_auth_hash(key: DCPEKey, approximation_factor: float, encrypted_result: EncryptResult) -> bool:
    """Verify the authentication hash of an encrypted vector."""
    expected_hash = _compute_auth_hash(key, approximation_factor, encrypted_result.iv, encrypted_result.ciphertext)
    # Use secrets.compare_digest for constant-time comparison to prevent timing attacks
    return secrets.compare_digest(
        expected_hash.get_bytes(), 
        encrypted_result.auth_hash.get_bytes()
    )

def encrypt_vector(key: DCPEKey, message: List[float], approximation_factor: float = 1.0) -> EncryptResult:
    """
    Encrypt a vector using DCPE algorithm.
    """
    # Input validation
    if not isinstance(message, list):
        raise ValueError("Message must be a list")
    if not all(isinstance(x, (int, float)) for x in message):
        raise ValueError("All message elements must be numbers")
    
    shuffled_message = shuffle(key, message)
    
    message_np = np.array(shuffled_message, dtype=np.float64)
    message_dimensionality = len(shuffled_message)
    
    # Generate random IV (nonce) for ChaCha20. 12 bytes is standard.
    iv = secrets.token_bytes(12)
    
    if message_dimensionality == 0:
        ciphertext = []
    else:
        noise_vector = _generate_noise_vector(key, iv, approximation_factor, message_dimensionality)
        ciphertext_np = key.scaling_factor * message_np + noise_vector
        
        if not np.all(np.isfinite(ciphertext_np)):
            raise ValueError("Embedding or approximation factor too large, causing overflow")
        
        ciphertext = ciphertext_np.tolist()
    
    auth_hash = _compute_auth_hash(key, approximation_factor, iv, ciphertext)
    
    return EncryptResult(ciphertext, iv, auth_hash)

def decrypt_vector(key: DCPEKey, encrypted_result: EncryptResult, approximation_factor: float = 1.0) -> List[float]:
    """
    Decrypt a vector using DCPE algorithm.
    """
    if not _verify_auth_hash(key, approximation_factor, encrypted_result):
        raise ValueError("Invalid authentication hash - data may be corrupted or tampered with")
    
    ciphertext = encrypted_result.ciphertext
    iv = encrypted_result.iv
    
    if len(ciphertext) == 0:
        return []
    
    ciphertext_np = np.array(ciphertext, dtype=np.float64) # Use float64 for better precision
    message_dimensionality = len(ciphertext)
    
    noise_vector = _generate_noise_vector(key, iv, approximation_factor, message_dimensionality)
    
    # Using float64 reduces numerical errors during the round trip.
    message_np = (ciphertext_np - noise_vector) / key.scaling_factor
    
    shuffled_message = message_np.tolist()
    original_message = unshuffle(key, shuffled_message)
    
    return original_message

# Testing

def generate_random_key(scaling_factor: float = 1.0) -> DCPEKey:
    return DCPEKey.generate_random(scaling_factor)

def test_dcpe_encryption():
    key = generate_random_key(scaling_factor=2.0)
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    approximation_factor = 1.0
    
    encrypted_result = encrypt_vector(key, test_vector, approximation_factor)
    print(f"Original vector: {test_vector}")
    print(f"Encrypted vector: {encrypted_result.ciphertext}")
    
    decrypted_vector = decrypt_vector(key, encrypted_result, approximation_factor)
    print(f"Decrypted vector: {decrypted_vector}")
    
    max_diff = np.max(np.abs(np.array(test_vector) - np.array(decrypted_vector)))
    print(f"Maximum difference: {max_diff}")
    
    # Allow for very small floating point inaccuracies
    if np.allclose(test_vector, decrypted_vector):
        print("✓ DCPE encryption/decryption test PASSED")
    else:
        print("✗ DCPE encryption/decryption test FAILED")
        
    # Test tampering
    print("\n--- Testing Tampering Detection ---")
    corrupted_ciphertext = encrypted_result.ciphertext.copy()
    corrupted_ciphertext[0] += 0.0001
    corrupted_result = EncryptResult(corrupted_ciphertext, encrypted_result.iv, encrypted_result.auth_hash)
    
    try:
        decrypt_vector(key, corrupted_result, approximation_factor)
    except ValueError as e:
        print(f"✓ Correctly caught tampering attempt: {e}")

if __name__ == "__main__":
    test_dcpe_encryption()