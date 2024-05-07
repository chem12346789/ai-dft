"""
Create a basic dataset class for torch DataLoader.
We use preprocessing to "normalize" the output using the input.
"""


class BasicDataset:
    """Documentation for a class."""

    def __init__(
        self,
        input_,
        middle_,
        output_,
    ):
        self.input = input_
        self.middle = middle_
        self.output = output_
        self.ids = list(input_.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "middle": self.middle[self.ids[idx]],
            "output": self.output[self.ids[idx]],
        }
