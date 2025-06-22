import numpy as np
import secrets
import hashlib
from typing import List, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

class DCPEKey:
    """Simplified DCPE encryption key combining encryption key bytes and scaling factor."""
    
    def __init__(self, key_bytes: bytes, scaling_factor: float):
        if not isinstance(key_bytes, bytes):
            raise ValueError("Key must be bytes")
        if len(key_bytes) != 32:
            raise ValueError("Key must be exactly 32 bytes")
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
        if len(hash_bytes) != 32:  # SHA-256 hash
            raise ValueError("Auth hash must be 32 bytes (SHA-256)")
        self.hash_bytes = hash_bytes
    
    def get_bytes(self) -> bytes:
        return self.hash_bytes

class EncryptResult:
    """Result of DCPE vector encryption containing ciphertext, IV, and auth hash."""
    
    def __init__(self, ciphertext: List[float], iv: bytes, auth_hash: AuthHash):
        self.ciphertext = ciphertext
        self.iv = iv
        self.auth_hash = auth_hash

def _create_rng_for_shuffle(key: DCPEKey) -> secrets.SystemRandom:
    """Create a deterministic RNG for shuffling based on the key."""
    # Use a fixed string combined with key for deterministic shuffling
    shuffle_seed_material = b"DCPE_SHUFFLE_" + key.key_bytes
    # Create a seed from the key material
    seed_hash = hashlib.sha256(shuffle_seed_material).digest()
    # Convert first 8 bytes to integer for seeding
    seed = int.from_bytes(seed_hash[:8], 'big')
    
    # Create and seed a random generator
    rng = secrets.SystemRandom()
    # Note: SystemRandom doesn't support seeding, so we'll use a different approach
    # We'll use the key-derived values directly for deterministic shuffling
    return key.key_bytes

def shuffle(key: DCPEKey, input_list: List[float]) -> List[float]:
    """Deterministically shuffle a list based on the encryption key."""
    if not input_list:
        return input_list
    
    # Create deterministic random values for sorting
    key_material = key.key_bytes
    indexed_input = list(enumerate(input_list))
    
    # Generate deterministic sort keys
    def get_sort_key(index_value_pair):
        index, _ = index_value_pair
        # Create deterministic value for this index
        hash_input = key_material + index.to_bytes(4, 'big')
        hash_result = hashlib.sha256(hash_input).digest()
        return int.from_bytes(hash_result[:8], 'big')
    
    # Sort by deterministic random values
    shuffled_indexed = sorted(indexed_input, key=get_sort_key)
    
    # Extract just the values
    return [value for _, value in shuffled_indexed]

def unshuffle(key: DCPEKey, input_list: List[float]) -> List[float]:
    """Reverse the deterministic shuffle operation."""
    if not input_list:
        return input_list
    
    # Create the same deterministic sort keys
    key_material = key.key_bytes
    indexed_input = list(enumerate(input_list))
    
    def get_sort_key(index_value_pair):
        index, _ = index_value_pair
        hash_input = key_material + index.to_bytes(4, 'big')
        hash_result = hashlib.sha256(hash_input).digest()
        return int.from_bytes(hash_result[:8], 'big')
    
    # Sort by the same deterministic values
    shuffled_indexed = sorted(indexed_input, key=get_sort_key)
    
    # Now we need to reverse this - create mapping from shuffled position to original
    unshuffled_list = [0.0] * len(input_list)
    for shuffled_pos, (original_pos, value) in enumerate(shuffled_indexed):
        unshuffled_list[original_pos] = input_list[shuffled_pos]
    
    return unshuffled_list

def _sample_normal_vector(coin_rng: secrets.SystemRandom, dimensionality: int) -> np.ndarray:
    """Sample a vector from multivariate normal distribution."""
    return np.array([coin_rng.gauss(0, 1) for _ in range(dimensionality)])

def _sample_uniform_point(coin_rng: secrets.SystemRandom) -> float:
    """Sample a uniform random point between 0 and 1."""
    return coin_rng.random()

def _generate_noise_vector(key: DCPEKey, iv: bytes, approximation_factor: float, dimensionality: int) -> np.ndarray:
    """Generate deterministic noise vector for DCPE encryption."""
    # Create deterministic RNG from key and IV
    seed_material = key.key_bytes + iv
    seed_hash = hashlib.sha256(seed_material).digest()
    
    # Create a deterministic random number generator
    # Note: We'll simulate this by using the hash to generate deterministic values
    coin_rng = secrets.SystemRandom()
    
    # Generate normal vector (we'll use key+iv+index for deterministic values)
    normal_vector = np.zeros(dimensionality)
    for i in range(dimensionality):
        # Create deterministic normal value for each dimension
        dim_seed = seed_material + i.to_bytes(4, 'big') + b"normal"
        dim_hash = hashlib.sha256(dim_seed).digest()
        # Convert to float in normal distribution (approximate)
        val = int.from_bytes(dim_hash[:8], 'big') / (2**64)
        # Convert uniform [0,1] to approximate normal using Box-Muller-like transformation
        normal_vector[i] = np.sqrt(-2 * np.log(max(val, 1e-10))) * np.cos(2 * np.pi * val)
    
    # Generate uniform point
    uniform_seed = seed_material + b"uniform"
    uniform_hash = hashlib.sha256(uniform_seed).digest()
    uniform_point = int.from_bytes(uniform_hash[:8], 'big') / (2**64)
    
    # Calculate noise parameters
    d_dimensional_ball_radius = key.scaling_factor / 4.0 * approximation_factor
    uniform_point_in_ball = d_dimensional_ball_radius * (uniform_point ** (1.0 / dimensionality))
    
    # Normalize and scale
    norm = np.linalg.norm(normal_vector)
    if norm > 0:
        normalized_vector = normal_vector * uniform_point_in_ball / norm
    else:
        normalized_vector = np.zeros(dimensionality)
    
    return normalized_vector

def _compute_auth_hash(key: DCPEKey, approximation_factor: float, iv: bytes, encrypted_embedding: List[float]) -> AuthHash:
    """Compute authentication hash for encrypted vector."""
    hasher = hashlib.sha256()
    
    # Add key
    hasher.update(key.key_bytes)
    
    # Add scaling factor
    hasher.update(key.scaling_factor.hex().encode())
    
    # Add approximation factor
    hasher.update(str(approximation_factor).encode())
    
    # Add IV
    hasher.update(iv)
    
    # Add each embedding value
    for value in encrypted_embedding:
        hasher.update(str(value).encode())
    
    return AuthHash(hasher.digest())

def _verify_auth_hash(key: DCPEKey, approximation_factor: float, encrypted_result: EncryptResult) -> bool:
    """Verify the authentication hash of an encrypted vector."""
    expected_hash = _compute_auth_hash(key, approximation_factor, encrypted_result.iv, encrypted_result.ciphertext)
    return expected_hash.get_bytes() == encrypted_result.auth_hash.get_bytes()

def encrypt_vector(key: DCPEKey, message: List[float], approximation_factor: float = 1.0) -> EncryptResult:
    """
    Encrypt a vector using DCPE algorithm.
    
    Args:
        key: DCPE encryption key
        message: Vector to encrypt (list of floats)
        approximation_factor: Controls noise magnitude (default 1.0)
    
    Returns:
        EncryptResult containing encrypted vector, IV, and auth hash
    """
    # Input validation
    if not isinstance(message, list):
        raise ValueError("Message must be a list")
    if not all(isinstance(x, (int, float)) for x in message):
        raise ValueError("All message elements must be numbers")
    
    # Shuffle the message for additional security
    shuffled_message = shuffle(key, message)
    
    # Convert to numpy array
    message_np = np.array(shuffled_message, dtype=np.float32)
    message_dimensionality = len(shuffled_message)
    
    # Generate random IV
    iv = secrets.token_bytes(12)
    
    # Handle empty vector case
    if message_dimensionality == 0:
        ciphertext = []
    else:
        # Generate noise vector
        noise_vector = _generate_noise_vector(key, iv, approximation_factor, message_dimensionality)
        
        # Encrypt: scaled_message + noise
        ciphertext_np = key.scaling_factor * message_np + noise_vector
        
        # Check for overflow
        if not np.all(np.isfinite(ciphertext_np)):
            raise ValueError("Embedding or approximation factor too large, causing overflow")
        
        ciphertext = ciphertext_np.tolist()
    
    # Compute authentication hash
    auth_hash = _compute_auth_hash(key, approximation_factor, iv, ciphertext)
    
    return EncryptResult(ciphertext, iv, auth_hash)

def decrypt_vector(key: DCPEKey, encrypted_result: EncryptResult, approximation_factor: float = 1.0) -> List[float]:
    """
    Decrypt a vector using DCPE algorithm.
    
    Args:
        key: DCPE encryption key
        encrypted_result: Result from encrypt_vector
        approximation_factor: Same value used during encryption
    
    Returns:
        Decrypted vector (list of floats)
    """
    # Verify authentication hash
    if not _verify_auth_hash(key, approximation_factor, encrypted_result):
        raise ValueError("Invalid authentication hash - data may be corrupted")
    
    ciphertext = encrypted_result.ciphertext
    iv = encrypted_result.iv
    
    # Handle empty vector case
    if len(ciphertext) == 0:
        return []
    
    # Convert to numpy array
    ciphertext_np = np.array(ciphertext, dtype=np.float32)
    message_dimensionality = len(ciphertext)
    
    # Regenerate the same noise vector
    noise_vector = _generate_noise_vector(key, iv, approximation_factor, message_dimensionality)
    
    # Decrypt: (ciphertext - noise) / scaling_factor
    message_np = (ciphertext_np - noise_vector) / key.scaling_factor
    
    # Convert back to list
    shuffled_message = message_np.tolist()
    
    # Unshuffle to get original order
    original_message = unshuffle(key, shuffled_message)
    
    return original_message

# Convenience functions for easy testing
def generate_random_key(scaling_factor: float = 1.0) -> DCPEKey:
    """Generate a random DCPE key for testing."""
    return DCPEKey.generate_random(scaling_factor)

def test_dcpe_encryption():
    """Simple test function to verify DCPE encryption/decryption works."""
    # Generate test data
    key = generate_random_key(scaling_factor=2.0)
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    approximation_factor = 1.0
    
    # Encrypt
    encrypted_result = encrypt_vector(key, test_vector, approximation_factor)
    print(f"Original vector: {test_vector}")
    print(f"Encrypted vector: {encrypted_result.ciphertext}")
    
    # Decrypt
    decrypted_vector = decrypt_vector(key, encrypted_result, approximation_factor)
    print(f"Decrypted vector: {decrypted_vector}")
    
    # Check if decryption is close to original (allowing for small numerical errors)
    max_diff = max(abs(orig - decr) for orig, decr in zip(test_vector, decrypted_vector))
    print(f"Maximum difference: {max_diff}")
    
    if max_diff < 1e-6:
        print("✓ DCPE encryption/decryption test PASSED")
    else:
        print("✗ DCPE encryption/decryption test FAILED")

if __name__ == "__main__":
    test_dcpe_encryption()