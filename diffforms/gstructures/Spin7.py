# TODO: Write Spin(7) version of SU(2) structure equations here, including natural operators that appear at this level

def GetCayleyForm(frame : list[DifferentialFormMul]) -> DifferentialFormMul:

     assert(len(frame)==8)

     return frame[0]*frame[1]*frame[2]*frame[3] - frame[0]*frame[1]*frame[4]*frame[5] - frame[0]*frame[1]*frame[6]*frame[7] - frame[0]*frame[2]*frame[4]*frame[6] + frame[0]*frame[2]*frame[5]*frame[7] - frame[0]*frame[3]*frame[4]*frame[7] - frame[0]*frame[3]*frame[5]*frame[6] + frame[4]*frame[5]*frame[6]*frame[7] - frame[2]*frame[3]*frame[6]*frame[7] - frame[2]*frame[3]*frame[4]*frame[5] - frame[1]*frame[3]*frame[5]*frame[7] + frame[1]*frame[3]*frame[4]*frame[6] - frame[1]*frame[2]*frame[5]*frame[6] - frame[1]*frame[2]*frame[4]*frame[7]
