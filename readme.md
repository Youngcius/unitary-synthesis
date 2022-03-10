# Two-Qubit Gate Decomposition & Reck Decomposition

>Date: 2011-11-27

## Necessary dependencies
- numpy, scipy: for linear algebra calculation
- matplotlib, qiskit: for circuit object visualization

## Main functions

- `decompose_gate`: any two-qubit gate decomposition (a high-level interface)

- `cnot_decomp`: decompose an arbitrary two-qubit gate into several single-qubit gates and at most three CNOT gates (based on KAK decomposition)

  Step 1: decompose into $ \left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right)$

  ```txt
  ---B0------------------A0---
          | exp(-iH) |
  ---B1------------------A1---
  ```

  Step 2: calculate parameterized gates $e^{-iH}$ with three CNOT gates

  ```txt
  ---B0---@---U0---@---V0---@---W--------A0---
          |        |        |
  ---B1---X---U1---X---V1---X---W^dag----A1---
  ```

- `abc_decomp`: for controlled-U gate decomposition

- `tensor_product_decomp`: when a two-qubit gate is in form of tensor product of two single-qubit gate

- `reck_decomp`: decompose an arbitrary U(N) operator into $\frac{N(N-1)}{2}$ U(2) operators *(currently only supports real unitary matrix)*

## Other Utility functions

- `params_zyz`: ZYZ decomposition of 2*2 unitary operator (standard U3 operator)
  $$
  U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)
  $$
  
- `params_u3`: Obtain the global phase "p" appended to the standard U3 operator
  
  $$
  U = e^{i p} U3(\theta, \phi, \lambda)
  $$
  
  where the U3 operator is in form of
  
  $$
  \begin{align}
  U3(\theta, \phi, \lambda) = e^{i\frac{\phi+\lambda}{2}}R_z(\phi)R_y(\theta)R_z(\lambda)
  = 
      \begin{pmatrix}
          \cos(\frac{\theta}{2})          & -e^{i\lambda}\sin(\frac{\theta}{2}) \\
          e^{i\phi}\sin(\frac{\theta}{2}) & e^{i(\phi+\lambda)}\cos(\frac{\theta}{2})
          \end{pmatrix}
  \end{align}
  $$
  
- `params_abc`: ABC decomposition of 2*2 unitary operator, based on the ZYZ decomposition algorithm
  $$
  \begin{align}
  U &= e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)\\
  &= e^{i\alpha}[R_z(\phi)R_y(\frac{\theta}{2})]X[R_y(-\frac{\theta}{2})R_z(-\frac{\phi+\lambda}{2})]X[R_z(\frac{\lambda-\phi}{2})]\\
  &=e^{i\alpha}AXBXC
  \end{align}
  $$
  

## How to use

See the `examples.ipynb` file.

## Reference

[1] Vidal, G. and C. M. Dawson (2004). "Universal quantum circuit for two-qubit transformations with three controlled-NOT gates." Physical Review A 69(1).

[2] Tucci, R. (2005). "An Introduction to Cartan's KAK Decomposition for QC Programmers."

[3] Kraus, B. and J. I. Cirac (2001). "Optimal creation of entanglement using a two-qubit gate." Physical Review A 63(6).

[4] Liu, F.-j. (2012). New Kronecker product decompositions and its applications.

[5] Vatan, F. and C. Williams (2004). "Optimal quantum circuits for general two-qubit gates." Physical Review A 69(3).

[6] Berman, A. and R. J. Plemmons (1994). Nonnegative matrices in the mathematical sciences, SIAM.
