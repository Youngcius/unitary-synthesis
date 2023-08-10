import cirq

# X, Y, Z, H, S, T, CNOT, CZ, SWAP, ISWAP, CZPowGate()
circ = cirq.testing.random_circuit(6, 12, 0.7)
print(circ)
# print(circ.to_qasm())
