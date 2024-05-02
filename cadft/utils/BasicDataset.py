class BasicDataset:
    """Documentation for a class."""

    def __init__(
        self,
        input,
        middle,
        output,
    ):
        self.input = input
        self.middle = middle
        self.output = output
        self.ids = list(input.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "middle": self.middle[self.ids[idx]],
            "output": self.output[self.ids[idx]],
        }
