import hashlib
import json
import time
from ecdsa import SECP256k1, VerifyingKey, SigningKey


# Define a simple blockchain class
class Blockchain:
    def __init__(self, authority_private_key):
        self.chain = []
        self.pending_certificates = []
        self.create_block(previous_hash='1', proof=100)  # Initialize the chain with a genesis block
        self.authority_private_key = authority_private_key
        self.authority_public_key = self.get_public_key_from_private(self.authority_private_key)

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'proof': proof,
            'previous_hash': previous_hash,
            'certificates': self.pending_certificates
        }
        self.pending_certificates = []  # Reset the list of pending certificates
        self.chain.append(block)
        return block

    def add_certificate(self, certificate_data):
        certificate_hash = self.hash_certificate(certificate_data)
        signature = self.sign_certificate(certificate_hash)
        certificate = {
            'data': certificate_data,
            'hash': certificate_hash,
            'signature': signature,
            'issuer': self.authority_public_key
        }
        self.pending_certificates.append(certificate)
        return certificate

    def sign_certificate(self, data_hash):
        signing_key = SigningKey.from_string(self.authority_private_key, curve=SECP256k1)
        signature = signing_key.sign(data_hash.encode())
        return signature.hex()

    def get_public_key_from_private(self, private_key):
        signing_key = SigningKey.from_string(private_key, curve=SECP256k1)
        public_key = signing_key.get_verifying_key()
        return public_key.to_string().hex()

    def hash_certificate(self, certificate_data):
        return hashlib.sha256(json.dumps(certificate_data, sort_keys=True).encode()).hexdigest()

    def validate_certificate(self, certificate):
        signature = bytes.fromhex(certificate['signature'])
        data_hash = certificate['hash']
        public_key = bytes.fromhex(certificate['issuer'])
        verifying_key = VerifyingKey.from_string(public_key, curve=SECP256k1)
        try:
            verifying_key.verify(signature, data_hash.encode())
            return True
        except:
            return False

    def proof_of_authority(self):
        last_block = self.chain[-1]
        last_proof = last_block['proof']
        last_hash = self.hash_block(last_block)
        proof = self.find_proof_of_authority(last_proof, last_hash)
        return proof

    def find_proof_of_authority(self, last_proof, last_hash):
        proof = 0
        while not self.is_valid_proof(last_proof, last_hash, proof):
            proof += 1
        return proof

    def is_valid_proof(self, last_proof, last_hash, proof):
        guess = f'{last_proof}{last_hash}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == '0000'  # Simplified PoA condition

    def hash_block(self, block):
        return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()

    def get_chain(self):
        return self.chain