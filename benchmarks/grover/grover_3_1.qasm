OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
x q[0];
h q[1];
h q[1];
x q[1];
h q[2];
cz q[0],q[2];
x q[0];
h q[0];
x q[0];
h q[2];
x q[2];
h q[2];
h q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
t q[2];
cx q[1],q[2];
t q[1];
tdg q[2];
cx q[0],q[2];
cx q[0],q[1];
t q[0];
tdg q[1];
cx q[0],q[1];
x q[0];
h q[0];
x q[1];
h q[1];
t q[2];
h q[2];
h q[2];
x q[2];
h q[2];
