# Unitary Synthesis (UniSys SDK)

> Latest update date: 2023 Jan.

## Dependencies
- numpy, scipy: for linear algebra calculation
- matplotlib, qiskit: for circuit object visualization

## Main functions

**Gate decomposition examples:**

- Fixed gate decomposition
  - SWAP 
  - CH
  - CCX
  - ...
- Universal gate decomposition
  - Tensor product decomposition
  - ABC decomposition (2-qubit controlled-U gate)
  - KAK decomposition (arbitrary 2-qubit gate)
  - Quantum Shannon decomposition (arbitrary unitary gate)
  - m-control n-target CU gate decomposition
- Continuous-variable unitary synthesis
  - Reck decomposition

**State preparation examples:**
- Arbitrary two-qubit state preparation
- Arbitrary three-qubit state preparation
  

## How to use

See the [example.ipynb](./example.ipynb) and [tests](./tests) for more details.
