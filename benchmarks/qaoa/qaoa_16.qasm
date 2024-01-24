// Generated from Cirq v1.0.0

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), q(8), q(9), q(10), q(11), q(12), q(13), q(14), q(15)]
qreg q[16];


h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
cx q[0],q[15];
cx q[4],q[5];
rz(pi*-0.2420283006) q[15];
rz(pi*-0.7947294816) q[5];
cx q[0],q[15];
cx q[4],q[5];
cx q[0],q[13];
cx q[5],q[6];
rz(pi*0.3367678945) q[13];
rz(pi*-0.9018057388) q[6];
cx q[0],q[13];
cx q[5],q[6];
cx q[0],q[9];
cx q[5],q[7];
cx q[6],q[8];
rz(pi*-0.9413605542) q[9];
rz(pi*0.5845986037) q[7];
rz(pi*0.0374331818) q[8];
cx q[0],q[9];
cx q[5],q[7];
cx q[6],q[8];
cx q[0],q[1];
cx q[7],q[8];
rx(pi*-0.9053209255) q[5];
rz(pi*0.2718007187) q[1];
rz(pi*0.5763743472) q[8];
cx q[0],q[1];
cx q[7],q[8];
cx q[0],q[3];
rz(pi*-0.9356041301) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.8367949356) q[0];
rz(pi*0.4895613103) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.0541739955) q[15];
rz(pi*0.3067297427) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.7564912891) q[12];
rz(pi*0.9921726547) q[15];
rz(pi*0.1475482273) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.0044326706) q[3];
rz(pi*0.0852718516) q[11];
rz(pi*0.5387946741) q[14];
rz(pi*0.3996681495) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.0726945502) q[2];
rz(pi*-0.8664511135) q[9];
rz(pi*0.3223357347) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.8174073156) q[1];
rx(pi*-0.3726620999) q[4];
rz(pi*-0.1482646116) q[13];
rz(pi*-0.1768615536) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.6179860893) q[5];
cx q[8],q[9];
rx(pi*-0.5166287255) q[6];
rx(pi*-0.8089407169) q[7];
cx q[4],q[5];
rz(pi*-0.037947449) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.730919703) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.6367423147) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.9495152844) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.357362201) q[13];
rz(pi*-0.6261925021) q[12];
rx(pi*-0.4136950627) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.5235001886) q[8];
rz(pi*0.6910659931) q[13];
rz(pi*-0.1654178782) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.4661883704) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.6155821726) q[9];
rx(pi*0.7899565758) q[10];
cx q[6],q[8];
rz(pi*0.9780690148) q[13];
rz(pi*-0.5268003766) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.8651027297) q[8];
cx q[14],q[15];
rx(pi*-0.9135542158) q[11];
rx(pi*-0.3961063275) q[12];
rx(pi*0.9611643972) q[13];
cx q[7],q[8];
rz(pi*0.8336646659) q[15];
cx q[14],q[15];
rx(pi*0.0790096451) q[14];
rx(pi*0.2526187234) q[15];
cx q[0],q[15];
rz(pi*-0.9889091832) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.0301811131) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.9766570692) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.249628945) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.8059236827) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.9881143126) q[0];
rz(pi*-0.0761824769) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.926008932) q[15];
rz(pi*-0.5835034065) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.3163387729) q[12];
rz(pi*-0.1132645964) q[15];
rz(pi*-0.1789604292) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*-0.6759683259) q[3];
rz(pi*0.5978454664) q[11];
rz(pi*0.4312025503) q[14];
rz(pi*0.9349886136) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.0677719634) q[2];
rz(pi*0.5976926624) q[9];
rz(pi*0.301500733) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.2685617481) q[1];
rx(pi*0.1948662167) q[4];
rz(pi*0.0041422007) q[13];
rz(pi*0.9860665222) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.9845249717) q[5];
cx q[8],q[9];
rx(pi*0.2641009896) q[6];
rx(pi*-0.9476067895) q[7];
cx q[4],q[5];
rz(pi*-0.5270752076) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.7502490677) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.2514156353) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.272847364) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.5719761702) q[13];
rz(pi*-0.5350404288) q[12];
rx(pi*-0.1864422131) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.7751869209) q[8];
rz(pi*-0.7891082678) q[13];
rz(pi*-0.398779729) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.0799198704) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.9677627392) q[9];
rx(pi*-0.7460839379) q[10];
cx q[6],q[8];
rz(pi*0.2688845358) q[13];
rz(pi*-0.4375304371) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.5490732794) q[8];
cx q[14],q[15];
rx(pi*0.5543249231) q[11];
rx(pi*-0.9082095356) q[12];
rx(pi*0.4219973872) q[13];
cx q[7],q[8];
rz(pi*-0.275446478) q[15];
cx q[14],q[15];
rx(pi*0.942092281) q[14];
rx(pi*0.7433658663) q[15];
cx q[0],q[15];
rz(pi*0.4203233026) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.917019486) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.1403733242) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.7457578286) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.2880846641) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.9929355781) q[0];
rz(pi*0.8595273058) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.7024446875) q[15];
rz(pi*-0.7521539802) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.8800580299) q[12];
rz(pi*0.1929737967) q[15];
rz(pi*0.4423687321) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*-0.6032686219) q[3];
rz(pi*0.6654323946) q[11];
rz(pi*-0.9672150382) q[14];
rz(pi*-0.8303554451) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.8549099971) q[2];
rz(pi*0.6921096764) q[9];
rz(pi*-0.5490031791) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.0382219123) q[1];
rx(pi*-0.8958177312) q[4];
rz(pi*0.1362064276) q[13];
rz(pi*0.144293536) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.4192464571) q[5];
cx q[8],q[9];
rx(pi*-0.2552070388) q[6];
rx(pi*0.7143061157) q[7];
cx q[4],q[5];
rz(pi*0.3219035901) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.5099339229) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.4035092134) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.7652031256) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.162746282) q[13];
rz(pi*0.8647013231) q[12];
rx(pi*0.4933952616) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.9467777689) q[8];
rz(pi*-0.0938221509) q[13];
rz(pi*0.174987495) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.3978932836) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.8402984595) q[9];
rx(pi*0.361805998) q[10];
cx q[6],q[8];
rz(pi*0.8965047432) q[13];
rz(pi*0.1120695075) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.8156278053) q[8];
cx q[14],q[15];
rx(pi*0.8084519881) q[11];
rx(pi*0.2150581416) q[12];
rx(pi*0.6239066249) q[13];
cx q[7],q[8];
rz(pi*0.0011228417) q[15];
cx q[14],q[15];
rx(pi*-0.3289122529) q[14];
rx(pi*-0.3008675439) q[15];
cx q[0],q[15];
rz(pi*-0.2202515393) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.5095941631) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.2614176511) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.515560387) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.8753367136) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.0859775651) q[0];
rz(pi*0.8160221673) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.3024053678) q[15];
rz(pi*-0.327320942) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.2692761405) q[12];
rz(pi*-0.3458002148) q[15];
rz(pi*0.6446076294) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.1614264268) q[3];
rz(pi*-0.4523155767) q[11];
rz(pi*0.7645522024) q[14];
rz(pi*0.9186904505) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.824264243) q[2];
rz(pi*-0.5877697426) q[9];
rz(pi*-0.1549132938) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.3879694057) q[1];
rx(pi*-0.5346272422) q[4];
rz(pi*-0.7094725321) q[13];
rz(pi*0.2058643934) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.5639291209) q[5];
cx q[8],q[9];
rx(pi*0.5555380351) q[6];
rx(pi*-0.5991973701) q[7];
cx q[4],q[5];
rz(pi*-0.2716251005) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.4403959631) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.1291406852) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.8657832963) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.6173285586) q[13];
rz(pi*-0.5689891054) q[12];
rx(pi*-0.4061969679) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.6411484394) q[8];
rz(pi*0.3538117193) q[13];
rz(pi*-0.4439528125) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.3712972927) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.0701302906) q[9];
rx(pi*0.5595333242) q[10];
cx q[6],q[8];
rz(pi*0.4835208443) q[13];
rz(pi*0.1194757913) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.9131638181) q[8];
cx q[14],q[15];
rx(pi*-0.52504356) q[11];
rx(pi*-0.3348394606) q[12];
rx(pi*0.9073942386) q[13];
cx q[7],q[8];
rz(pi*-0.3303271743) q[15];
cx q[14],q[15];
rx(pi*0.3156301463) q[14];
rx(pi*0.545755661) q[15];
cx q[0],q[15];
rz(pi*0.3767486864) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.5913917643) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.0586225031) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.6179277454) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.3500702538) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.5877369274) q[0];
rz(pi*-0.9879442287) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.8251845147) q[15];
rz(pi*-0.4596474651) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.3064105596) q[12];
rz(pi*-0.2791525611) q[15];
rz(pi*-0.1575998857) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.7267067031) q[3];
rz(pi*0.8887310792) q[11];
rz(pi*-0.5786947449) q[14];
rz(pi*0.6915050146) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.2763654689) q[2];
rz(pi*-0.0176190381) q[9];
rz(pi*-0.0874588019) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.6698671143) q[1];
rx(pi*0.0188034547) q[4];
rz(pi*0.8194293241) q[13];
rz(pi*0.4142301205) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.7289605551) q[5];
cx q[8],q[9];
rx(pi*0.9005032501) q[6];
rx(pi*0.6319321793) q[7];
cx q[4],q[5];
rz(pi*-0.0322219219) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.7818236055) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.1115578773) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.8131458625) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.9273533113) q[13];
rz(pi*-0.3344927659) q[12];
rx(pi*0.2253645656) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.3540521144) q[8];
rz(pi*-0.918633619) q[13];
rz(pi*0.8942390798) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.6749322169) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.9441964905) q[9];
rx(pi*0.9747021957) q[10];
cx q[6],q[8];
rz(pi*0.2353199542) q[13];
rz(pi*-0.2622503166) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.3234330804) q[8];
cx q[14],q[15];
rx(pi*-0.1826797325) q[11];
rx(pi*0.3118462058) q[12];
rx(pi*-0.1886936031) q[13];
cx q[7],q[8];
rz(pi*0.2239540781) q[15];
cx q[14],q[15];
rx(pi*-0.4853037885) q[14];
rx(pi*-0.834694648) q[15];
cx q[0],q[15];
rz(pi*-0.4727793078) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.457040293) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.2027218406) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.6302279379) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.9076368066) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.3043077476) q[0];
rz(pi*-0.7942402292) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.2504170663) q[15];
rz(pi*0.7366294203) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.1166052239) q[12];
rz(pi*-0.4390460387) q[15];
rz(pi*0.8361940319) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*-0.7307730053) q[3];
rz(pi*-0.152963902) q[11];
rz(pi*-0.9588476852) q[14];
rz(pi*-0.4461964199) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.549160409) q[2];
rz(pi*-0.2560164342) q[9];
rz(pi*0.0469750967) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.5340402066) q[1];
rx(pi*-0.6688800585) q[4];
rz(pi*-0.1794685649) q[13];
rz(pi*0.886401117) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.9998362248) q[5];
cx q[8],q[9];
rx(pi*-0.5224331877) q[6];
rx(pi*0.4095570951) q[7];
cx q[4],q[5];
rz(pi*-0.5097388169) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.8389449327) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.9736803374) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.1689928983) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.9517031884) q[13];
rz(pi*0.8491037696) q[12];
rx(pi*0.8656850652) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.3009629452) q[8];
rz(pi*0.4187713848) q[13];
rz(pi*-0.065339454) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.4892309243) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.44515208) q[9];
rx(pi*0.9978368119) q[10];
cx q[6],q[8];
rz(pi*-0.2497817038) q[13];
rz(pi*0.0857208494) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.2153918578) q[8];
cx q[14],q[15];
rx(pi*-0.9187677509) q[11];
rx(pi*0.2916450434) q[12];
rx(pi*-0.92260083) q[13];
cx q[7],q[8];
rz(pi*0.7178336752) q[15];
cx q[14],q[15];
rx(pi*0.5204205158) q[14];
rx(pi*-0.539820085) q[15];
cx q[0],q[15];
rz(pi*-0.8203362659) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.2968994238) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.4652024346) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.3561906298) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.8961981057) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.8447647092) q[0];
rz(pi*-0.4113861089) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.0978233076) q[15];
rz(pi*0.2243587234) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.4257934205) q[12];
rz(pi*0.9764298873) q[15];
rz(pi*-0.555685875) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.1901477527) q[3];
rz(pi*0.6210269125) q[11];
rz(pi*0.8051130772) q[14];
rz(pi*0.9611946841) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.3203782919) q[2];
rz(pi*-0.7377697897) q[9];
rz(pi*0.7654259694) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.3932385319) q[1];
rx(pi*-0.1173517291) q[4];
rz(pi*-0.5743370029) q[13];
rz(pi*0.7030961028) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.0753268881) q[5];
cx q[8],q[9];
rx(pi*-0.2048718967) q[6];
rx(pi*-0.0444439032) q[7];
cx q[4],q[5];
rz(pi*-0.7447755516) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.5954651904) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.7877307357) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.2558436978) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.0069840552) q[13];
rz(pi*-0.3887072238) q[12];
rx(pi*0.0414366398) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.2343721771) q[8];
rz(pi*-0.1478086893) q[13];
rz(pi*0.8336975705) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.923336786) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.1905210281) q[9];
rx(pi*0.9849568719) q[10];
cx q[6],q[8];
rz(pi*0.0352469215) q[13];
rz(pi*0.6080527367) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.7238241909) q[8];
cx q[14],q[15];
rx(pi*-0.8022974309) q[11];
rx(pi*-0.5587933644) q[12];
rx(pi*-0.3546897378) q[13];
cx q[7],q[8];
rz(pi*0.7153035746) q[15];
cx q[14],q[15];
rx(pi*-0.7045543127) q[14];
rx(pi*-0.4315615309) q[15];
cx q[0],q[15];
rz(pi*0.5584905857) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.0457840018) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.9320927277) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.9652451704) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.2320129551) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.294865026) q[0];
rz(pi*-0.8821210428) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.3223375436) q[15];
rz(pi*0.454159901) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.2432612586) q[12];
rz(pi*0.3422532089) q[15];
rz(pi*0.0497324429) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.0538812775) q[3];
rz(pi*-0.7286534057) q[11];
rz(pi*-0.504973693) q[14];
rz(pi*0.4336067283) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.5200916135) q[2];
rz(pi*0.1273291858) q[9];
rz(pi*-0.2802653021) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.9252157825) q[1];
rx(pi*0.7515424211) q[4];
rz(pi*0.0929580433) q[13];
rz(pi*0.1351483251) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.0697040271) q[5];
cx q[8],q[9];
rx(pi*-0.9299336629) q[6];
rx(pi*-0.7127980622) q[7];
cx q[4],q[5];
rz(pi*-0.6483434691) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.7563209445) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.0207527407) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.0514230825) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*0.5138916731) q[13];
rz(pi*0.6341981575) q[12];
rx(pi*0.934873206) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.5912091773) q[8];
rz(pi*-0.7797896071) q[13];
rz(pi*-0.6650367177) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.1075032695) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.0160479023) q[9];
rx(pi*-0.1162414571) q[10];
cx q[6],q[8];
rz(pi*0.0681529844) q[13];
rz(pi*-0.2285130351) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.0988261176) q[8];
cx q[14],q[15];
rx(pi*-0.3631304372) q[11];
rx(pi*-0.4309016035) q[12];
rx(pi*0.931772625) q[13];
cx q[7],q[8];
rz(pi*-0.502752465) q[15];
cx q[14],q[15];
rx(pi*-0.1340613384) q[14];
rx(pi*0.7680060654) q[15];
cx q[0],q[15];
rz(pi*0.2963262464) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.7168552925) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.7048990895) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.9126240556) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.3958844724) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.2466460055) q[0];
rz(pi*0.6107938708) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.4662557908) q[15];
rz(pi*-0.9181844139) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.2104536719) q[12];
rz(pi*0.0322216711) q[15];
rz(pi*-0.514075627) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*-0.6159397126) q[3];
rz(pi*0.4347082678) q[11];
rz(pi*0.5853027215) q[14];
rz(pi*-0.1300285781) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.333054238) q[2];
rz(pi*0.4315008228) q[9];
rz(pi*-0.1944256614) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.7922315351) q[1];
rx(pi*-0.049064425) q[4];
rz(pi*0.3267855146) q[13];
rz(pi*-0.9449141407) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.3175352121) q[5];
cx q[8],q[9];
rx(pi*-0.9366621387) q[6];
rx(pi*-0.6965400999) q[7];
cx q[4],q[5];
rz(pi*-0.9361640248) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.8863611424) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.402719602) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.8895556644) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*0.4151622371) q[13];
rz(pi*0.753409363) q[12];
rx(pi*0.968256576) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.4028416313) q[8];
rz(pi*0.9198782674) q[13];
rz(pi*-0.0638806652) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.2426567509) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.8836139288) q[9];
rx(pi*0.8176835923) q[10];
cx q[6],q[8];
rz(pi*0.2518130234) q[13];
rz(pi*-0.0856365446) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.5489302314) q[8];
cx q[14],q[15];
rx(pi*-0.6759983184) q[11];
rx(pi*0.9622355466) q[12];
rx(pi*0.5014950503) q[13];
cx q[7],q[8];
rz(pi*-0.5541075267) q[15];
cx q[14],q[15];
rx(pi*0.0799541659) q[14];
rx(pi*0.8634057659) q[15];
cx q[0],q[15];
rz(pi*0.7612142844) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.217367015) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.312686392) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.2947702911) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.3460636284) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.5858999002) q[0];
rz(pi*-0.6412196507) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.0663802451) q[15];
rz(pi*-0.0777242589) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.4734379274) q[12];
rz(pi*0.369782931) q[15];
rz(pi*0.9917221569) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.6206296905) q[3];
rz(pi*-0.2898697473) q[11];
rz(pi*-0.3275402105) q[14];
rz(pi*-0.6079810685) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.5293284304) q[2];
rz(pi*0.9082879378) q[9];
rz(pi*-0.8036320028) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.3111031013) q[1];
rx(pi*-0.6733246173) q[4];
rz(pi*-0.9660170013) q[13];
rz(pi*0.6025535682) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.8100569157) q[5];
cx q[8],q[9];
rx(pi*-0.5443958684) q[6];
rx(pi*0.1788308663) q[7];
cx q[4],q[5];
rz(pi*0.7509196586) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.5864337022) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.0920203712) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.4570163271) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.2689587634) q[13];
rz(pi*-0.7660589725) q[12];
rx(pi*-0.5813320185) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.1752315106) q[8];
rz(pi*-0.4515499771) q[13];
rz(pi*-0.7685109281) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.031560453) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.9347237713) q[9];
rx(pi*0.3153348862) q[10];
cx q[6],q[8];
rz(pi*0.9052053968) q[13];
rz(pi*0.6172522295) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.5482721396) q[8];
cx q[14],q[15];
rx(pi*0.1698085326) q[11];
rx(pi*0.0375451573) q[12];
rx(pi*0.5293150762) q[13];
cx q[7],q[8];
rz(pi*-0.6704412788) q[15];
cx q[14],q[15];
rx(pi*-0.7878894788) q[14];
rx(pi*-0.9958161976) q[15];
cx q[0],q[15];
rz(pi*0.9049777356) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.002684641) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*-0.3433292423) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.2638934798) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.6076866322) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*0.7582848768) q[0];
rz(pi*-0.2352595752) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.540338348) q[15];
rz(pi*-0.0377433448) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.1190759958) q[12];
rz(pi*-0.0663005796) q[15];
rz(pi*0.8872294899) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.9892207737) q[3];
rz(pi*0.6881549255) q[11];
rz(pi*-0.4713440414) q[14];
rz(pi*-0.1128073961) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*0.4667907973) q[2];
rz(pi*-0.8475918719) q[9];
rz(pi*-0.8056807886) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.8721229018) q[1];
rx(pi*0.0023795616) q[4];
rz(pi*-0.3232457776) q[13];
rz(pi*-0.0479467847) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*0.1272559661) q[5];
cx q[8],q[9];
rx(pi*0.1892871658) q[6];
rx(pi*0.2482999595) q[7];
cx q[4],q[5];
rz(pi*0.7407410089) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.5808160818) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.9915635003) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.4964458586) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.5603281) q[13];
rz(pi*0.6950046211) q[12];
rx(pi*0.5507295257) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.3361454747) q[8];
rz(pi*0.2233427544) q[13];
rz(pi*0.8904732699) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.0477613783) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.6547765157) q[9];
rx(pi*0.7974253849) q[10];
cx q[6],q[8];
rz(pi*-0.4198271499) q[13];
rz(pi*0.4540854879) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.2375235564) q[8];
cx q[14],q[15];
rx(pi*0.241982735) q[11];
rx(pi*-0.9128625909) q[12];
rx(pi*0.3680821294) q[13];
cx q[7],q[8];
rz(pi*-0.9699677032) q[15];
cx q[14],q[15];
rx(pi*-0.6078319067) q[14];
rx(pi*-0.9453184374) q[15];
cx q[0],q[15];
rz(pi*0.1019065505) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.6266272772) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.7198822966) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.7929581497) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.3260855714) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.4400000979) q[0];
rz(pi*0.4201504489) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.4109660119) q[15];
rz(pi*0.0385607223) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.9427279974) q[12];
rz(pi*0.3886297724) q[15];
rz(pi*-0.3228355955) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.5087832098) q[3];
rz(pi*-0.4426250599) q[11];
rz(pi*-0.5106804259) q[14];
rz(pi*0.7733563316) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.3515567511) q[2];
rz(pi*-0.8600356212) q[9];
rz(pi*0.4946518119) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.3153204821) q[1];
rx(pi*-0.7729818432) q[4];
rz(pi*0.5379173899) q[13];
rz(pi*0.0026485326) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.7712303354) q[5];
cx q[8],q[9];
rx(pi*0.1718039493) q[6];
rx(pi*0.6707773743) q[7];
cx q[4],q[5];
rz(pi*0.1942506797) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.840660866) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.5121200583) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.8951636703) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*0.0741595876) q[13];
rz(pi*0.894134981) q[12];
rx(pi*0.1010593809) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.1382486957) q[8];
rz(pi*0.7955054884) q[13];
rz(pi*0.8307090133) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.6823277315) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.2499289064) q[9];
rx(pi*0.1088242631) q[10];
cx q[6],q[8];
rz(pi*0.5090366801) q[13];
rz(pi*-0.507357992) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.1601536578) q[8];
cx q[14],q[15];
rx(pi*0.9513425327) q[11];
rx(pi*0.5109487828) q[12];
rx(pi*0.0896264988) q[13];
cx q[7],q[8];
rz(pi*-0.229457108) q[15];
cx q[14],q[15];
rx(pi*-0.6519358189) q[14];
rx(pi*0.8082284337) q[15];
cx q[0],q[15];
rz(pi*-0.5883244346) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.3000865136) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.8729437059) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.5528407396) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.5481529185) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.2256095902) q[0];
rz(pi*0.7036378208) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.6553100519) q[15];
rz(pi*0.9758722062) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.296593294) q[12];
rz(pi*0.6706862018) q[15];
rz(pi*0.027358652) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.6952939904) q[3];
rz(pi*-0.4698074226) q[11];
rz(pi*0.7987832099) q[14];
rz(pi*-0.8948393231) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.9717123888) q[2];
rz(pi*-0.7452230555) q[9];
rz(pi*-0.3388357944) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.1117191692) q[1];
rx(pi*0.8438397233) q[4];
rz(pi*-0.6826417131) q[13];
rz(pi*-0.5075141626) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.3769276601) q[5];
cx q[8],q[9];
rx(pi*-0.4639577664) q[6];
rx(pi*0.9804780004) q[7];
cx q[4],q[5];
rz(pi*-0.5893004505) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.3022141519) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.3696517002) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*0.2940391903) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.0277766646) q[13];
rz(pi*-0.7995710753) q[12];
rx(pi*-0.8218080642) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*-0.23361194) q[8];
rz(pi*-0.35018071) q[13];
rz(pi*0.089526742) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.5005075934) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*0.3873107886) q[9];
rx(pi*0.3799050945) q[10];
cx q[6],q[8];
rz(pi*-0.3059496939) q[13];
rz(pi*-0.2178083877) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.6073071108) q[8];
cx q[14],q[15];
rx(pi*-0.1313818707) q[11];
rx(pi*-0.6016836853) q[12];
rx(pi*0.9331587518) q[13];
cx q[7],q[8];
rz(pi*-0.3789825164) q[15];
cx q[14],q[15];
rx(pi*-0.8726182814) q[14];
rx(pi*-0.0297012244) q[15];
cx q[0],q[15];
rz(pi*-0.5585385804) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.4120517355) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.6570546444) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.265468874) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*-0.8333034606) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.5626357207) q[0];
rz(pi*-0.6073819898) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.7207468205) q[15];
rz(pi*-0.837602005) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.9540577001) q[12];
rz(pi*0.4469311837) q[15];
rz(pi*0.8363198366) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.6246524786) q[3];
rz(pi*-0.4640356788) q[11];
rz(pi*-0.1671267752) q[14];
rz(pi*0.8829339948) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.2131593265) q[2];
rz(pi*0.3508179754) q[9];
rz(pi*0.0064948542) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*-0.0889607154) q[1];
rx(pi*0.57111351) q[4];
rz(pi*-0.5404728053) q[13];
rz(pi*0.9197991277) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.385490695) q[5];
cx q[8],q[9];
rx(pi*0.9040214499) q[6];
rx(pi*0.0549133548) q[7];
cx q[4],q[5];
rz(pi*-0.0141725448) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*0.437531671) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*0.5032299257) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.7976360918) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*-0.0520162373) q[13];
rz(pi*0.1682779623) q[12];
rx(pi*-0.5534590513) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.1928079257) q[8];
rz(pi*0.1750803201) q[13];
rz(pi*0.9597726158) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*0.0323319139) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.1898864574) q[9];
rx(pi*0.2990019114) q[10];
cx q[6],q[8];
rz(pi*0.3368662484) q[13];
rz(pi*-0.5204610569) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*0.4896090539) q[8];
cx q[14],q[15];
rx(pi*0.742652607) q[11];
rx(pi*0.3478719336) q[12];
rx(pi*0.9401970876) q[13];
cx q[7],q[8];
rz(pi*-0.9696046817) q[15];
cx q[14],q[15];
rx(pi*0.4022444993) q[14];
rx(pi*0.6434414729) q[15];
cx q[0],q[15];
rz(pi*-0.9099208324) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*0.345397028) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.3095052889) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*-0.796507898) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.6847749921) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.8375983002) q[0];
rz(pi*0.2283448078) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*-0.8033438192) q[15];
rz(pi*-0.9604878188) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*0.1889342374) q[12];
rz(pi*-0.2688654376) q[15];
rz(pi*-0.341441733) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*-0.8785207541) q[3];
rz(pi*-0.0431683033) q[11];
rz(pi*0.2397021549) q[14];
rz(pi*0.5022424842) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.2326545017) q[2];
rz(pi*-0.5334128602) q[9];
rz(pi*0.5172492951) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.6915964473) q[1];
rx(pi*0.7928513431) q[4];
rz(pi*0.1155973207) q[13];
rz(pi*0.8063554445) q[9];
cx q[4],q[5];
cx q[6],q[13];
cx q[7],q[9];
rz(pi*-0.1012962434) q[5];
cx q[8],q[9];
rx(pi*-0.4637511394) q[6];
rx(pi*-0.6110043209) q[7];
cx q[4],q[5];
rz(pi*-0.2619222679) q[9];
cx q[5],q[6];
cx q[8],q[9];
rz(pi*-0.2437581374) q[6];
cx q[8],q[10];
cx q[5],q[6];
rz(pi*-0.1426730553) q[10];
cx q[5],q[7];
cx q[8],q[10];
rz(pi*-0.2357816141) q[7];
cx q[8],q[13];
cx q[10],q[12];
cx q[5],q[7];
rz(pi*0.4655349848) q[13];
rz(pi*0.1157398018) q[12];
rx(pi*-0.2713166409) q[5];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.9350021198) q[8];
rz(pi*0.3252728373) q[13];
rz(pi*-0.299720736) q[14];
cx q[6],q[8];
cx q[9],q[13];
cx q[10],q[14];
rz(pi*-0.8977473989) q[8];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.7749198277) q[9];
rx(pi*0.4443264843) q[10];
cx q[6],q[8];
rz(pi*-0.6092953069) q[13];
rz(pi*-0.632385256) q[14];
cx q[7],q[8];
cx q[11],q[13];
cx q[12],q[14];
rz(pi*-0.9685091254) q[8];
cx q[14],q[15];
rx(pi*0.8641774861) q[11];
rx(pi*0.3360025906) q[12];
rx(pi*0.7174532189) q[13];
cx q[7],q[8];
rz(pi*-0.8368334191) q[15];
cx q[14],q[15];
rx(pi*-0.5151057962) q[14];
rx(pi*0.3478559572) q[15];
cx q[0],q[15];
rz(pi*0.4017426799) q[15];
cx q[0],q[15];
cx q[0],q[13];
rz(pi*-0.0833349747) q[13];
cx q[0],q[13];
cx q[0],q[9];
rz(pi*0.7410912412) q[9];
cx q[0],q[9];
cx q[0],q[1];
rz(pi*0.3887722003) q[1];
cx q[0],q[1];
cx q[0],q[3];
rz(pi*0.7897555823) q[3];
cx q[0],q[3];
cx q[1],q[3];
rx(pi*-0.2954461059) q[0];
rz(pi*0.5064087065) q[3];
cx q[1],q[3];
cx q[1],q[15];
cx q[2],q[3];
rz(pi*0.0405808426) q[15];
rz(pi*0.0702828) q[3];
cx q[1],q[15];
cx q[2],q[3];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
rz(pi*-0.002623568) q[12];
rz(pi*-0.1540535326) q[15];
rz(pi*-0.7618606032) q[10];
cx q[1],q[12];
cx q[2],q[15];
cx q[3],q[10];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
rx(pi*0.4543820101) q[3];
rz(pi*-0.0925447466) q[11];
rz(pi*-0.6849328078) q[14];
rz(pi*-0.9201738929) q[12];
cx q[1],q[11];
cx q[2],q[14];
cx q[4],q[12];
cx q[1],q[9];
cx q[4],q[13];
rx(pi*-0.203724229) q[2];
rz(pi*-0.9567062743) q[9];
rz(pi*0.9731597782) q[13];
cx q[1],q[9];
cx q[4],q[13];
cx q[6],q[13];
cx q[7],q[9];
rx(pi*0.6041513341) q[1];
rx(pi*0.1622460178) q[4];
rz(pi*-0.1466553321) q[13];
rz(pi*-0.9398127333) q[9];
cx q[6],q[13];
cx q[7],q[9];
cx q[8],q[9];
rx(pi*-0.83998696) q[6];
rx(pi*-0.7677492471) q[7];
rz(pi*-0.3218015337) q[9];
cx q[8],q[9];
cx q[8],q[10];
rz(pi*0.6419378955) q[10];
cx q[8],q[10];
cx q[8],q[13];
cx q[10],q[12];
rz(pi*-0.0823578401) q[13];
rz(pi*-0.6735599366) q[12];
cx q[8],q[13];
cx q[10],q[12];
cx q[9],q[13];
cx q[10],q[14];
rx(pi*0.7791174466) q[8];
rz(pi*-0.9703188416) q[13];
rz(pi*0.4798454439) q[14];
cx q[9],q[13];
cx q[10],q[14];
cx q[11],q[13];
cx q[12],q[14];
rx(pi*-0.09531898) q[9];
rx(pi*0.9880090779) q[10];
rz(pi*0.4765874743) q[13];
rz(pi*0.5090458128) q[14];
cx q[11],q[13];
cx q[12],q[14];
cx q[14],q[15];
rx(pi*-0.272206103) q[11];
rx(pi*-0.5000913907) q[12];
rx(pi*-0.2989213604) q[13];
rz(pi*-0.2966612476) q[15];
cx q[14],q[15];
rx(pi*-0.3138278061) q[14];
rx(pi*0.2747134633) q[15];
