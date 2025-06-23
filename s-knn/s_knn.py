import numpy as np
import secrets
import hashlib
from typing import List, Tuple

class SKNNKey:
    """Simplified S-KNN encryption key using AES key bytes."""
    
    def __init__(self, key_bytes: bytes):
        if not isinstance(key_bytes, bytes):
            raise ValueError("Key must be bytes")
        if len(key_bytes) != 32:
            raise ValueError("Key must be exactly 32 bytes")
        
        self.key_bytes = key_bytes
    
    @classmethod
    def generate_random(cls):
        """Generate a random S-KNN key."""
        return cls(secrets.token_bytes(32))

class EncryptResult:
    """Result of S-KNN vector encryption containing transformed vector."""
    
    def __init__(self, ciphertext: List[float]):
        self.ciphertext = ciphertext

def _generate_projection_matrix(key: SKNNKey, vector_dimension: int) -> np.ndarray:
    """
    Generate a deterministic random ORTHOGONAL matrix from the key.
    An orthogonal matrix represents a pure rotation/reflection.
    """
    # Create deterministic seed from key
    seed_hash = hashlib.sha256(key.key_bytes).digest()
    seed = int.from_bytes(seed_hash[:4], 'big') % (2**32)
    
    # Set numpy random seed for reproducible matrix generation
    np.random.seed(seed)
    
    # Generate a random square matrix
    random_matrix = np.random.randn(vector_dimension, vector_dimension)
    
    # Use QR decomposition to get a true orthogonal matrix (Q).
    # This is the standard way to generate a random rotation matrix.
    q_matrix, _ = np.linalg.qr(random_matrix)

    
    return q_matrix

def encrypt_vector(key: SKNNKey, message: List[float]) -> EncryptResult:
    """
    Encrypt a vector by applying a secret rotation.
    """
    # Input validation
    if not isinstance(message, list):
        raise ValueError("Message must be a list")
    if not all(isinstance(x, (int, float)) for x in message):
        raise ValueError("All message elements must be numbers")
    if len(message) == 0:
        return EncryptResult([])
    
    vector = np.array(message, dtype=np.float32)
    vector_dimension = len(message)
    
    # Generate the deterministic orthogonal matrix (rotation)
    rotation_matrix = _generate_projection_matrix(key, vector_dimension)
    
    # Apply the rotation
    transformed_vector = rotation_matrix @ vector
    
    if not np.all(np.isfinite(transformed_vector)):
        raise ValueError("Vector transformation resulted in non-finite values")
    
    ciphertext = transformed_vector.astype('float32').tolist()
    return EncryptResult(ciphertext)

def decrypt_vector(key: SKNNKey, encrypted_result: EncryptResult) -> List[float]:
    """
    Decrypt a vector by applying the inverse rotation.
    """
    ciphertext = encrypted_result.ciphertext
    if len(ciphertext) == 0:
        return []
        
    encrypted_vector = np.array(ciphertext, dtype=np.float32)
    vector_dimension = len(ciphertext)
    
    # Regenerate the same orthogonal matrix
    rotation_matrix = _generate_projection_matrix(key, vector_dimension)
    
    # For an orthogonal matrix, the inverse is simply its transpose.
    # This is numerically exact, stable, and much faster than pseudo-inverse.
    inverse_rotation_matrix = rotation_matrix.T

    
    # Apply inverse transformation to recover original vector
    original_vector = inverse_rotation_matrix @ encrypted_vector
    
    decrypted_message = original_vector.astype('float32').tolist()
    return decrypted_message


def generate_random_key() -> SKNNKey:
    return SKNNKey.generate_random()

def test_sknn_encryption():
    key = generate_random_key()
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    encrypted_result = encrypt_vector(key, test_vector)
    print(f"Original vector: {test_vector}")
    print(f"Encrypted vector: {encrypted_result.ciphertext}")
    
    decrypted_vector = decrypt_vector(key, encrypted_result)
    print(f"Decrypted vector: {decrypted_vector}")
    
    max_diff = max(abs(orig - decr) for orig, decr in zip(test_vector, decrypted_vector))
    print(f"Maximum difference: {max_diff}")
    
    # With an orthogonal matrix and transpose, decryption is nearly perfect.
    # We can use a much tighter tolerance.
    if max_diff < 1e-6:
        print("✓ S-KNN encryption/decryption test PASSED")
    else:
        print("✗ S-KNN encryption/decryption test FAILED")
    
    test_similarity_preservation(key)

def test_similarity_preservation(key: SKNNKey):
    """Test that S-KNN preserves similarity relationships between vectors."""
    vector1 = [1.0, 0.0, 0.0, 0.0, 0.0]
    vector2 = [0.9, 0.1, 0.0, 0.0, 0.0]  # Similar to vector1
    vector3 = [0.0, 0.0, 1.0, 0.0, 0.0]  # Different from vector1

    enc1 = encrypt_vector(key, vector1)
    enc2 = encrypt_vector(key, vector2)
    enc3 = encrypt_vector(key, vector3)

    def cosine_similarity(a, b):
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    orig_sim_12 = cosine_similarity(vector1, vector2)
    orig_sim_13 = cosine_similarity(vector1, vector3)
    enc_sim_12 = cosine_similarity(enc1.ciphertext, enc2.ciphertext)
    enc_sim_13 = cosine_similarity(enc1.ciphertext, enc3.ciphertext)
    
    print(f"\nSimilarity Preservation Test:")
    print(f"Original similarity (v1, v2): {orig_sim_12:.4f}")
    print(f"Encrypted similarity (v1, v2): {enc_sim_12:.4f}")
    print(f"Original similarity (v1, v3): {orig_sim_13:.4f}")
    print(f"Encrypted similarity (v1, v3): {enc_sim_13:.4f}")

    if np.isclose(orig_sim_12, enc_sim_12, atol=1e-6) and np.isclose(orig_sim_13, enc_sim_13, atol=1e-6):
        print("✓ Similarity values perfectly preserved (within float precision)")
    else:
        print("✗ Similarity values NOT perfectly preserved")

if __name__ == "__main__":
    test_sknn_encryption()