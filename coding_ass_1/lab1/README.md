# Lab 1 - Probability Review

These instructions are simplified and sufficient only to get up and running. 
Please refer to the prob-review-lab.pdf for full instructions.

## Setup

If Using Conda:

```bash
conda create -n pr python=3.9 -y
conda activate pr
pip install matplotlib pytest
```

If using system python on Ubuntu:

```bash
sudo apt-get install python3
sudo apt-get install python3-pip
pip3 install matplotlib pytest
```

## Code Execution

Note: if using conda, ensure that your environment is active, and in the following instead of 'python3' just use 'python'.

Instuctions for running Part 1:
```python3 probability_review/part1.py```

Instuctions for running Part 2:
```python3 probability_review/part2.py```

Instuctions for running Part 3:
```python3 probability_review/part3.py```

## Test Execution

Pytest is a very versatile testing framework.
Many options exist to customize the output.
If you feel like more detailed output would be useful, try adding '-v' after 'pytest', and if that isn't enough feel free to look more into [pytest's documentation](https://docs.pytest.org).

Instructions for running test for Part 1:
```pytest test/test_part1.py```

Instructions for running test for Part 2:
```pytest test/test_part2.py```

Instructions for running test for Part 3:
```pytest test/test_part3.py```