"""
Create a basic dataset class for torch DataLoader.
"""


class BasicDataset:
    """Documentation for a class."""

    def __init__(
        self,
        input_,
        output_,
    ):
        self.input = input_
        self.output = output_
        self.ids = list(input_.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "output": self.output[self.ids[idx]],
        }
