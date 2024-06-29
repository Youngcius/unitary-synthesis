import cirq
from unisys import Circuit
from unisys.utils import passes


def test_contract_1q_gate_on_dag():
    qasm_fname = '../../benchmarks/alu/alu-v2_33.qasm'
    circ = Circuit.from_qasm(fname=qasm_fname)
    dag = circ.to_dag()
    dag = passes.contract_1q_gates_on_dag(dag)

    cirq.testing.assert_allclose_up_to_global_phase(
        passes.dag_to_circuit(dag).unitary(),
        passes.dag_to_circuit(dag).unitary(),
        atol=1e-6
    )
