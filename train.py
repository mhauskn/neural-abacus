"""
Train a model to perform addition/subtraction/encoding.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from model import Model
from abacus import Soroban
from enum import Enum
import math

PAD_VALUE=-1

class Operation(Enum):
    ENCODE = 0
    ADD = 1
    SUBTRACT = 2

class AbacusDataset(Dataset):
    """Tests the model's ability to add, subtract, and encode numbers using an Abacus."""
    def __init__(self, split: str) -> None:
        super().__init__()
        num = 100000
        rng = torch.Generator()
        rng.manual_seed(1337)
        pow1 = torch.randint(low=0, high=10, size=(num,))
        pow2 = torch.randint(low=0, high=10, size=(num,))
        # We need double precision here to get the required number of decimal places.
        perm1 = torch.rand(num, generator=rng, dtype=torch.double) * 10**pow1
        perm2 = torch.rand(num, generator=rng, dtype=torch.double) * 10**pow2
        self.ops = torch.randint(low=0, high=3, size=(num,))
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes1 = perm1[:num_test] if split == 'test' else perm1[num_test:]
        self.ixes2 = perm2[:num_test] if split == 'test' else perm2[num_test:]
        self.data = []
        for ix1, ix2, op in tqdm(zip(self.ixes1, self.ixes2, self.ops), desc='Generating Dataset', total=len(self.ixes1)):
            data_point = generate_data_sequence(ix1, ix2, Operation(op.item()))
            self.data.append(data_point)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_vocab_size():
        return 10 # Digits 0-9


def _encode(value: float) -> torch.Tensor:
    """Encode the value as a string '00009932000' then as a tensor of integers."""
    value = '{0:.3f}'.format(value).replace(".", "").zfill(13)
    return torch.tensor([int(s) for s in value], dtype=torch.long)


def generate_data_sequence(v1:float, v2: float, op: Operation) -> list:
    """Generate data points and instructions."""
    ab = Soroban()
    if op == Operation.ENCODE:
        instrs = ab.get_instructions_for_setting_value(v1)
        numerical_input = _encode(v1)
    elif op == Operation.ADD:
        ab.value = v1
        instrs = ab.get_instructions_for_add(v2)
        ab.value = v1
        numerical_input = _encode(v2)
    elif op == Operation.SUBTRACT:
        if v1 < v2:
            tmp = v1
            v1 = v2
            v2 = tmp
        ab.value = v1
        instrs = ab.get_instructions_for_subtract(v2)
        ab.value = v1
        numerical_input = _encode(v2)
    else:
        raise ValueError(f"Unexpected Operation {op}")
    encoded_abs, encoded_instrs = [], []
    for instr in instrs:
        encoded_abs.append(ab.to_numpy())
        encoded_instrs.append(ab.encode_instruction(instr))
        instr()
    encoded_abs.append(ab.to_numpy())
    encoded_instrs.append([4,0,0]) # Encode stop instruction.
    data_point = {
        "operation": torch.tensor([op.value], dtype=torch.long),
        "numerical_inputs": numerical_input,
        "ab_states": torch.tensor(np.array(encoded_abs), dtype=torch.float32),
        "instructions": torch.tensor(encoded_instrs, dtype=torch.long)
    }
    return data_point


def visualize(model):
    """Visualize the model's decisions step-by-step."""
    model.eval()
    eval_dataset = AbacusDataset(split='test')
    ab = Soroban()
    
    def _setup(op, v1, v2):
        if op == Operation.ENCODE:
            numerical_input = v1
            ab.reset()
            expected_result = v1
            s = f"Encode {v1:.3f}"
        elif op == Operation.ADD:
            ab.value = v1
            numerical_input = v2
            expected_result = math.floor((v1 + v2) * 1000)/1000
            s = f"{v1:.3f} + {v2:.3f}"
        elif op == Operation.SUBTRACT:
            if v1 < v2:
                tmp = v1
                v1 = v2
                v2 = tmp
            ab.value = v1
            numerical_input = v2
            expected_result = math.floor((v1 - v2) * 1000)/1000
            s = f"{v1:.3f} - {v2:.3f}"
        return numerical_input, expected_result, s

    for v1, v2, op in zip(eval_dataset.ixes1, eval_dataset.ixes2, eval_dataset.ops):
        op = Operation(op.item())
        numerical_input, expected_result, s = _setup(op, v1, v2)
        
        print(f"Input an arithemtic expression, or press Enter for default: {s}")
        custom_value = input()
        if custom_value:
            if '+' in custom_value:
                v1 = float(custom_value.split('+')[0].strip())
                v2 = float(custom_value.split('+')[1].strip())
                op = Operation.ADD
            elif '-' in custom_value:
                v1 = float(custom_value.split('-')[0].strip())
                v2 = float(custom_value.split('-')[1].strip())
                op = Operation.SUBTRACT
            else:
                v1 = float(custom_value)
                op = Operation.ENCODE
            numerical_input, expected_result, s = _setup(op, v1, v2)

        numerical_input = _encode(numerical_input).unsqueeze(0).to(device)
        encoded_abs, prev_instrs = [],[]
        prev_instrs.append(np.array([0,0,0])) # Start with a dummy instruction
        for idx in range(40): # Should be able to finish in 40 steps
            print(ab)
            encoded_abs.append(ab.to_numpy())
            ab_reps = torch.tensor(np.array(encoded_abs), dtype=torch.float32).unsqueeze(0).to(device)
            prev_instrs_tt = torch.tensor(np.array(prev_instrs), dtype=torch.long).unsqueeze(0).to(device)
            ops = torch.tensor([op.value], dtype=torch.long).unsqueeze(0).to(device)
            decoded_instr = model.decode(ab_reps, numerical_input, prev_instrs_tt, ops)
            # Take the instruction from the final timestep
            decoded_instr = decoded_instr[:,-1,:].squeeze().cpu().numpy()
            if decoded_instr[0] == 4:
                print(f"\nStep {idx}. STOP")
                break
            prev_instrs.append(decoded_instr)
            decoded_str = ab.decode_instruction_str(decoded_instr)
            decoded_instr = ab.decode_instruction(decoded_instr)
            print(f"\nStep {idx}. {decoded_str}")
            decoded_instr() # Apply the instruction to change the abacus configuration.
        result = "Correct" if np.isclose(ab.value, expected_result, atol=1e-3) else "Incorrect"
        print(f"{result}: {s} = {expected_result:.3f}")


def collate_batch(batch: list[dict]):
    """Custom function to collate our dictionary-based batch items and pad them to length."""
    def collate(key):
        return pad_sequence([i[key] for i in batch], batch_first=True, padding_value=PAD_VALUE)
    return {k: collate(k) for k in batch[0].keys()}

def roll_instrs(instrs: torch.Tensor):
    """Creates previous instructions by rolling the current instructions by 1"""
    prev_instrs = torch.roll(instrs, shifts=1, dims=1) # Push the instructions back by 1
    prev_instrs[prev_instrs==PAD_VALUE] = 0 # Replace all -1s with 0s
    prev_instrs[:,0,:] = 0 # Ensure always starts with [0,0,0] instruction
    return prev_instrs

def train(model):
    """Train the model and save it."""
    dataset = AbacusDataset(split='train')
    eval_dataset = AbacusDataset(split='test')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(
            dataset=dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=64,
            num_workers=0,
            collate_fn=collate_batch,
        )
    for epoch in range(10):
        evaluate(model, eval_dataset)
        model.train()
        pbar = tqdm(loader)
        for data in pbar:
            ops = data["operation"].to(device)
            ab_reps = data["ab_states"].to(device)
            numerical_inputs = data["numerical_inputs"].to(device)
            instrs = data["instructions"].to(device)
            prev_instrs = roll_instrs(instrs)
            optimizer.zero_grad()
            _, loss = model.forward(ab_reps, numerical_inputs, prev_instrs, ops, targets=instrs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} Loss {loss:.3E}")
    evaluate(model, eval_dataset)
    torch.save(model.state_dict(), 'model.pth')


def evaluate(model, dataset=None):
    """Evaluate how well the model does on a heldout dataset."""
    with torch.no_grad():
        model.eval()
        if dataset is None:
            dataset = AbacusDataset(split='test')
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False, collate_fn=collate_batch)
        correct, total = 0, 0
        for b, item in enumerate(loader):
            ops = item["operation"].to(device)
            ab_reps = item["ab_states"].to(device)
            numerical_inputs = item["numerical_inputs"].to(device)
            instrs = item["instructions"]
            prev_instrs = roll_instrs(instrs).to(device)
            decoded_instr = model.decode(ab_reps, numerical_inputs, prev_instrs, ops)
            # Add -1s everywhere that the original instr has -1s so we don't penalize padding
            decoded_instr[instrs==PAD_VALUE] = PAD_VALUE
            invalid = torch.sum(torch.all(instrs == PAD_VALUE, dim=-1)) # Disqualify any -1 answers
            result = torch.all(instrs == decoded_instr.cpu(), dim=-1)
            correct += torch.sum(result) - invalid
            total += result.numel() - invalid
        print(f"Evaluation {correct}/{total} Correct = {100*correct/total:.2f}%")



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(vocab_size=AbacusDataset.get_vocab_size()).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:.3e} parameters')
    train(model)
    model.load_state_dict(torch.load('model.pth'))
    visualize(model)
