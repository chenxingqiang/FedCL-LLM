class FederatedLearning:
    def __init__(self, clients, server, model, rounds):
        self.clients = clients
        self.server = server
        self.model = model
        self.rounds = rounds

    def train(self):
        for r in range(self.rounds):
            # Local update for each client
            local_updates = [client.local_update(self.model) for client in self.clients]
            # Server aggregates updates
            self.model = self.server.aggregate(local_updates)
            print(f"Round {r} completed.")
        return self.model
