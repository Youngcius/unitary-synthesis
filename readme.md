# Unitary Synthesis (UniSys SDK)

> Latest update date: 2023 Jan.

## Dependencies
- numpy, scipy: for linear algebra calculation
- matplotlib, qiskit: for circuit object visualization

## Main functions

- `decompose_gate`: any two-qubit gate decomposition (a high-level interface)

- `cnot_decomp`: decompose an arbitrary two-qubit gate into several single-qubit gates and at most three CNOT gates (based on KAK decomposition)

  Step 1: decompose an arbitrary two-qubit gate into $ \left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right)$

  ```txt
       ┌──────────┐     
  ──B0─┤          ├─A0──
       │ exp(-iH) │     
  ──B1─┤          ├─A1──
       └──────────┘    
  ``` 
  Step 2: calculate parameterized gates $e^{-iH}$ with three CNOT gates

  ```txt
  ──B0────●────U0────●────V0────●────W─────A0── 
          │          │          │               
  ──B1────X────U1────X────V1────X────W†────A1── 
  ```

- `abc_decomp`: for controlled-U gate decomposition

- `tensor_product_decomp`: when a two-qubit gate is in form of tensor product of two single-qubit gates

- `reck_decomp`: decompose an arbitrary U(N) operator into $\frac{N(N-1)}{2}$ U(2) operators *(currently only supports real unitary matrix)*

## Other Utility functions

- `params_zyz`: ZYZ decomposition of 2*2 unitary operator (standard U3 operator)
  $$
  U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)
  $$
  Remind that for arbitrary unitary gate,
  $$
  \begin{align}
  U &= e^{i\alpha}R_z(\phi)R_y(\theta)R_z(\lambda)\\
  &=e^{i(\alpha - \frac{\phi + \lambda}{2})} U3(\theta, \varphi,\lambda)\\
  &= e^{i(\alpha - \frac{\phi + \lambda}{2})} e^{i\frac{\phi+\lambda}{2}}R_z(\phi)R_y(\theta)R_z(\lambda)
  \end{align}
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
  
- `params_abc`: ABC decomposition of 2*2 unitary operator, based on ZYZ decomposition 

  $$
  \begin{align}
  U &= e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)\\
  &= e^{i\alpha}[R_z(\phi)R_y(\frac{\theta}{2})]X[R_y(-\frac{\theta}{2})R_z(-\frac{\phi+\lambda}{2})]X[R_z(\frac{\lambda-\phi}{2})]\\
  &=e^{i\alpha}AXBXC
  \end{align}
  $$
  

## How to use

See the [examples.ipynb](./src/examples.ipynb) file.
