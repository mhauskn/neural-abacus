import random
import numpy as np
from typing import Union
from functools import partial

class Soroban:
    """A soroban is a Japenese abacus with one row of upper beads 5 and 4 rows of lower beads. 
    
    Top Row is represented as an int in [0-1] indicating how many beads are active.
    Bottom rows are represented as an int in [0-4] indicating how many beads are active.

    Column Index   9 8 7 6 5 4 3  2  1  0
    Exponents      6 5 4 3 2 1 0 -1 -2 -3
    """
    def __init__(self) -> None:
        self.n_cols = 13
        self.col_exponents = [n for n in range(self.n_cols-4, -4, -1)]
        self.top = None
        self.bottom = None
        self.trace = None
        self.reset()

    def __str__(self) -> None:
        s = "_" * 3 * self.n_cols
        top_top = "  ".join(["o" if t == 0 else " " for t in self.top])
        bot_top = "  ".join(["o" if t == 1 else " " for t in self.top])
        divider = "-" * 3 * self.n_cols
        bot_0 = "  ".join([" " if t == 0 else "o" for t in self.bottom])
        bot_1 = "  ".join([" " if t == 1 else "o" for t in self.bottom])
        bot_2 = "  ".join([" " if t == 2 else "o" for t in self.bottom])
        bot_3 = "  ".join([" " if t == 3 else "o" for t in self.bottom])
        bot_4 = "  ".join([" " if t == 4 else "o" for t in self.bottom])
        bot = "=" * 3 * self.n_cols
        tail = "  ".join([str(self._col_val(col_idx)) for col_idx in range(self.n_cols)])
        return "\n".join([s, top_top, bot_top, divider, bot_0, bot_1, bot_2, bot_3, bot_4, bot, tail])

    def randomize(self) -> None:
        """Put the abacus into a random configuration. """
        self.top = [random.choice([0, 1]) for _ in range(self.n_cols)]
        self.bottom = [random.choice([0, 1, 2, 3, 4]) for _ in range(self.n_cols)]

    def reset(self) -> None:
        """Reset the abacus to the initial state. """
        self.top = np.zeros(self.n_cols, dtype=bool) #[0] * self.n_cols
        self.bottom = np.zeros(self.n_cols, dtype=np.int) #[0] * self.n_cols

    def push_bot(self, col: int, n: int = 1) -> None:
        """Moves n beads to active in the given column. No effect if all are already active."""
        self.bottom[col] = min(self.bottom[col] + n, 4)
        if self.trace is not None:
            self.trace.append(partial(self.push_bot, col, n))

    def pull_bot(self, col:int, n: int = 1) -> None:
        """Moves one bead to deactive state in the given column. No effect is already deactivated."""
        self.bottom[col] = max(self.bottom[col] - n, 0)
        if self.trace is not None:
            self.trace.append(partial(self.pull_bot, col, n))

    def push_top(self, col: int) -> None:
        self.top[col] = 1
        if self.trace is not None:
            self.trace.append(partial(self.push_top, col))

    def pull_top(self, col: int) -> None:
        self.top[col] = 0
        if self.trace is not None:
            self.trace.append(partial(self.pull_top, col))

    @property
    def value(self):
        """Get the value displayed by the Soroban."""
        return sum([10**self.col_exponents[col_idx] * self._col_val(col_idx) for col_idx in range(self.n_cols)])

    @value.setter
    def value(self, value):
        """Set the value displayed by the Soroban."""
        self.reset()
        s = '{0:.3f}'.format(value).zfill(self.n_cols)
        zeroes_col_idx = self.col_exponents.index(0)
        if '.' in s:
            decimal_idx = s.index('.')
            for col_idx, val in zip(range(zeroes_col_idx+1)[::-1], s[:decimal_idx][::-1]):
                self._set_col(col_idx, int(val))
            for col_idx, val in zip(range(zeroes_col_idx+1, self.n_cols), s[decimal_idx+1:]):
                self._set_col(col_idx, int(val))
        else: # Integer
            for col_idx, val in zip(range(zeroes_col_idx+1)[::-1], s[::-1]):
                self._set_col(col_idx, int(val))

    def _set_col(self, col: int, value: int):
        """Set the value of a particular column."""
        assert value >= 0 and value < 10
        if value >= 5:
            self.top[col] = 1
        self.bottom[col] = value % 5

    def _col_val(self, col_idx: int) -> int:
        """Get the value for a particular column."""
        return 5 * self.top[col_idx] + self.bottom[col_idx]

    def get_instructions_for_setting_value(self, value: Union[float, int]):
        """Returns a list of instructions required to set the desired value."""
        # I've been taught to set value from left to right e.g. most significant digit to least significant digit.
        s = '{0:.3f}'.format(value).replace(".", "").zfill(self.n_cols)
        instructions = []
        for col_idx, val in enumerate(s):
            val = int(val)
            if self._col_val(col_idx) != val:
                # Lets start with the top...
                if val >= 5:
                    if self.top[col_idx] == 0:
                        instructions.append(partial(self.push_top, col_idx))
                else:
                    if self.top[col_idx] == 1:
                        instructions.append(partial(self.pull_top, col_idx))
                # Now for the bottom...
                bot_diff = self.bottom[col_idx] - (val % 5)
                if bot_diff < 0:
                    instructions.append(partial(self.push_bot, col_idx, -bot_diff))
                elif bot_diff > 0:
                    instructions.append(partial(self.pull_bot, col_idx, bot_diff))
        return instructions

    def _add(self, col_idx: int, val: int):
        """Add a value in [0-9] to the column."""
        if val == 0:
            return
        curr_col_val = self._col_val(col_idx)
        # Need to subtract the complement and add one in the previous col
        if val + curr_col_val >= 10:
            complement = 10 - val
            # First, subtract the complement
            self._subtract(col_idx, complement)
            # Next add ten to the previous column
            if col_idx > 0:
                self._add(col_idx-1, 1)
        elif val >= 5: # Simple case - no carries or complements required.
            assert self.top[col_idx] == 0
            self.push_top(col=col_idx)
            if val != 5:
                self.push_bot(col=col_idx, n=val-5)
        elif self.bottom[col_idx] + val >= 5:
            # Add 5 and subtract the complement
            self.push_top(col=col_idx)
            self.pull_bot(col=col_idx, n=5-val)
        else: # Just add val to the bottom
            self.push_bot(col=col_idx, n=val)

    def _subtract(self, col_idx: int, val: int):
        """Subtract a value in [0-9] from the column."""
        if val == 0:
            return
        curr_col_val = self._col_val(col_idx)
        # Need to add the complement and subtract ten in the previous col
        if curr_col_val - val < 0:
            complement = 10 - val
            # First, add the complement
            self._add(col_idx, complement)
            # Next subtract ten from the previous column
            if col_idx > 0:
                self._subtract(col_idx-1, 1)
        elif val >= 5: # Simple case - no carries or complements required.
            assert self.top[col_idx] == 1
            self.pull_top(col=col_idx)
            if val != 5:
                self.pull_bot(col=col_idx, n=val-5)
        elif self.bottom[col_idx] - val < 0:
            # Add 5 and subtract the complement
            self.pull_top(col=col_idx)
            self.push_bot(col=col_idx, n=5-val)
        else: # Just subtract val from the bottom
            self.pull_bot(col=col_idx, n=val)

    def get_instructions_for_add(self, value: Union[float, int]) -> list[partial]:
        """Add the desired value and returns a list of instructions executed."""
        s = '{0:.3f}'.format(value).replace(".", "").zfill(self.n_cols)
        instructions = []
        self.trace = instructions
        for col_idx, val in enumerate(s):
            val = int(val)
            self._add(col_idx, val)
        self.trace = None
        return instructions

    def get_instructions_for_subtract(self, value: Union[float, int]) -> list[partial]:
        """Subtract the desired value and return a list of instructions executed."""
        s = '{0:.3f}'.format(value).replace(".", "").zfill(self.n_cols)
        instructions = []
        self.trace = instructions
        for col_idx, val in enumerate(s):
            val = int(val)
            self._subtract(col_idx, val)
        self.trace = None
        return instructions

    def encode_instruction(self, instr: partial) -> tuple:
        """Encodes a instruction as a tuple of ints (function_index, column_index, n)."""
        instr_idx = [self.pull_bot, self.push_bot, self.pull_top, self.push_top].index(instr.func)
        column_arg = instr.args[0]
        number_arg = instr.args[1] if len(instr.args) > 1 else 1
        return (instr_idx, column_arg, number_arg)

    def decode_instruction(self, instr: tuple[int]):
        """Decodes an instruction into a partial function"""
        fn = [self.pull_bot, self.push_bot, self.pull_top, self.push_top][instr[0]]
        arg0 = instr[1]
        arg1 = instr[2]
        if fn == self.push_top or fn == self.pull_top:
            return partial(fn, arg0)
        else:
            return partial(fn, arg0, arg1)

    def decode_instruction_str(self, instr: tuple[int]):
        """Decodes an instruction into an intepretable string"""
        fn = [self.pull_bot, self.push_bot, self.pull_top, self.push_top][instr[0]]
        arg0 = instr[1]
        arg1 = instr[2]
        z = "Remove" if (fn == self.pull_bot or fn == self.pull_bot) else "Add"
        if fn == self.push_top or fn == self.pull_top:
            return f"{z} top bead in column {arg0}"
        else:
            return f"{z} {arg1} bottom bead(s) in column {arg0}"

    def to_numpy(self) -> np.ndarray:
        """Returns a binary numpy encoding of the soroban's configuration."""
        arr = np.zeros((7, self.n_cols), dtype=bool)
        arr[0] = self.top == 0
        arr[1] = self.top == 1
        arr[2] = self.bottom != 0
        arr[3] = self.bottom != 1
        arr[4] = self.bottom != 2
        arr[5] = self.bottom != 3
        arr[6] = self.bottom != 4
        return arr


if __name__ == "__main__":
    ab = Soroban()
    # ab.push_bot(7, 3)
    # ab.push_top(6)
    # ab.push_bot(12, 2)

    for idx in range(1000):
        a = random.randint(0, 100000)
        b = random.randint(0, 100000)
        if a < b:
            tmp = a
            a = b
            b = tmp
        ab.value = a
        instrs = ab.get_instructions_for_add(b)
        ab.value = a
        for instr in instrs:
            instr()
        print(f"{a} + {b} = {ab.value}")

        assert ab.value == a + b

    # print(ab.get_instructions_for_setting_value(85349))

    # for instr in ab.get_instructions_for_setting_value(112):
    #     print(instr)
    #     instr()

    # print(ab)