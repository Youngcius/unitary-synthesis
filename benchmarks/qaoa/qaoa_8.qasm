// Generated from Cirq v1.3.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7)]
qreg q[8];


h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[6];
rz(pi*0.3929383712) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*0.3929383712) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*0.3929383712) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*0.3929383712) q[3];
rz(pi*0.3929383712) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*0.3929383712) q[7];
rz(pi*0.3929383712) q[4];
rz(pi*0.3929383712) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*-0.4277213301) q[0];
rx(pi*-0.4277213301) q[1];
rz(pi*0.3929383712) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*0.3929383712) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*-0.4277213301) q[2];
rz(pi*0.3929383712) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*0.3929383712) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*-0.4277213301) q[3];
rz(pi*0.3929383712) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*0.3929383712) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*-0.4277213301) q[4];
rz(pi*0.3929383712) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*-0.4277213301) q[5];
rz(pi*0.3929383712) q[7];
cx q[6],q[7];
rx(pi*-0.4277213301) q[6];
rx(pi*-0.4277213301) q[7];
cx q[0],q[6];
rz(pi*-0.5462970929) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*-0.5462970929) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*-0.5462970929) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*-0.5462970929) q[3];
rz(pi*-0.5462970929) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*-0.5462970929) q[7];
rz(pi*-0.5462970929) q[4];
rz(pi*-0.5462970929) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*0.1026295382) q[0];
rx(pi*0.1026295382) q[1];
rz(pi*-0.5462970929) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*-0.5462970929) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*0.1026295382) q[2];
rz(pi*-0.5462970929) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*-0.5462970929) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*0.1026295382) q[3];
rz(pi*-0.5462970929) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*-0.5462970929) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*0.1026295382) q[4];
rz(pi*-0.5462970929) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*0.1026295382) q[5];
rz(pi*-0.5462970929) q[7];
cx q[6],q[7];
rx(pi*0.1026295382) q[6];
rx(pi*0.1026295382) q[7];
cx q[0],q[6];
rz(pi*0.4389379396) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*0.4389379396) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*0.4389379396) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*0.4389379396) q[3];
rz(pi*0.4389379396) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*0.4389379396) q[7];
rz(pi*0.4389379396) q[4];
rz(pi*0.4389379396) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*-0.1537870798) q[0];
rx(pi*-0.1537870798) q[1];
rz(pi*0.4389379396) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*0.4389379396) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*-0.1537870798) q[2];
rz(pi*0.4389379396) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*0.4389379396) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*-0.1537870798) q[3];
rz(pi*0.4389379396) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*0.4389379396) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*-0.1537870798) q[4];
rz(pi*0.4389379396) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*-0.1537870798) q[5];
rz(pi*0.4389379396) q[7];
cx q[6],q[7];
rx(pi*-0.1537870798) q[6];
rx(pi*-0.1537870798) q[7];
cx q[0],q[6];
rz(pi*0.9615283968) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*0.9615283968) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*0.9615283968) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*0.9615283968) q[3];
rz(pi*0.9615283968) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*0.9615283968) q[7];
rz(pi*0.9615283968) q[4];
rz(pi*0.9615283968) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*0.3696594772) q[0];
rx(pi*0.3696594772) q[1];
rz(pi*0.9615283968) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*0.9615283968) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*0.3696594772) q[2];
rz(pi*0.9615283968) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*0.9615283968) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*0.3696594772) q[3];
rz(pi*0.9615283968) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*0.9615283968) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*0.3696594772) q[4];
rz(pi*0.9615283968) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*0.3696594772) q[5];
rz(pi*0.9615283968) q[7];
cx q[6],q[7];
rx(pi*0.3696594772) q[6];
rx(pi*0.3696594772) q[7];
cx q[0],q[6];
rz(pi*-0.038136197) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*-0.038136197) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*-0.038136197) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*-0.038136197) q[3];
rz(pi*-0.038136197) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*-0.038136197) q[7];
rz(pi*-0.038136197) q[4];
rz(pi*-0.038136197) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*-0.2157649636) q[0];
rx(pi*-0.2157649636) q[1];
rz(pi*-0.038136197) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*-0.038136197) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*-0.2157649636) q[2];
rz(pi*-0.038136197) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*-0.038136197) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*-0.2157649636) q[3];
rz(pi*-0.038136197) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*-0.038136197) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*-0.2157649636) q[4];
rz(pi*-0.038136197) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*-0.2157649636) q[5];
rz(pi*-0.038136197) q[7];
cx q[6],q[7];
rx(pi*-0.2157649636) q[6];
rx(pi*-0.2157649636) q[7];
cx q[0],q[6];
rz(pi*-0.3136439677) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*-0.3136439677) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*-0.3136439677) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*-0.3136439677) q[3];
rz(pi*-0.3136439677) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*-0.3136439677) q[7];
rz(pi*-0.3136439677) q[4];
rz(pi*-0.3136439677) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*0.4580994148) q[0];
rx(pi*0.4580994148) q[1];
rz(pi*-0.3136439677) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*-0.3136439677) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*0.4580994148) q[2];
rz(pi*-0.3136439677) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*-0.3136439677) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*0.4580994148) q[3];
rz(pi*-0.3136439677) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*-0.3136439677) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*0.4580994148) q[4];
rz(pi*-0.3136439677) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*0.4580994148) q[5];
rz(pi*-0.3136439677) q[7];
cx q[6],q[7];
rx(pi*0.4580994148) q[6];
rx(pi*0.4580994148) q[7];
cx q[0],q[6];
rz(pi*-0.1228555106) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*-0.1228555106) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*-0.1228555106) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*-0.1228555106) q[3];
rz(pi*-0.1228555106) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*-0.1228555106) q[7];
rz(pi*-0.1228555106) q[4];
rz(pi*-0.1228555106) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*-0.8806442068) q[0];
rx(pi*-0.8806442068) q[1];
rz(pi*-0.1228555106) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*-0.1228555106) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*-0.8806442068) q[2];
rz(pi*-0.1228555106) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*-0.1228555106) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*-0.8806442068) q[3];
rz(pi*-0.1228555106) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*-0.1228555106) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*-0.8806442068) q[4];
rz(pi*-0.1228555106) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*-0.8806442068) q[5];
rz(pi*-0.1228555106) q[7];
cx q[6],q[7];
rx(pi*-0.8806442068) q[6];
rx(pi*-0.8806442068) q[7];
cx q[0],q[6];
rz(pi*-0.2039114893) q[6];
cx q[0],q[6];
cx q[0],q[4];
rz(pi*-0.2039114893) q[4];
cx q[0],q[4];
cx q[0],q[1];
rz(pi*-0.2039114893) q[1];
cx q[0],q[1];
cx q[0],q[3];
cx q[1],q[5];
rz(pi*-0.2039114893) q[3];
rz(pi*-0.2039114893) q[5];
cx q[0],q[3];
cx q[1],q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
rz(pi*-0.2039114893) q[7];
rz(pi*-0.2039114893) q[4];
rz(pi*-0.2039114893) q[5];
cx q[0],q[7];
cx q[1],q[4];
cx q[2],q[5];
cx q[2],q[7];
rx(pi*0.4759908115) q[0];
rx(pi*0.4759908115) q[1];
rz(pi*-0.2039114893) q[7];
cx q[2],q[7];
cx q[2],q[3];
rz(pi*-0.2039114893) q[3];
cx q[2],q[3];
cx q[3],q[5];
rx(pi*0.4759908115) q[2];
rz(pi*-0.2039114893) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi*-0.2039114893) q[4];
cx q[3],q[4];
cx q[4],q[6];
rx(pi*0.4759908115) q[3];
rz(pi*-0.2039114893) q[6];
cx q[4],q[6];
cx q[4],q[7];
rz(pi*-0.2039114893) q[7];
cx q[4],q[7];
cx q[5],q[7];
rx(pi*0.4759908115) q[4];
rz(pi*-0.2039114893) q[7];
cx q[5],q[7];
cx q[6],q[7];
rx(pi*0.4759908115) q[5];
rz(pi*-0.2039114893) q[7];
cx q[6],q[7];
rx(pi*0.4759908115) q[6];
rx(pi*0.4759908115) q[7];
