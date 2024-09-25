import torch

class MemoryNetwork:
    def __init__(self, memory_size, vector_dim):
        self.memory = torch.zeros(memory_size, vector_dim)
    
    def write(self, key, value):
        """Write new knowledge to memory."""
        self.memory[key] = value

    def read(self, key):
        """Retrieve knowledge from memory."""
        return self.memory[key]
