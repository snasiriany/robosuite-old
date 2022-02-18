'''
Code extended directly from @sklaw: https://github.com/sklaw/obj_2_mujoco_msh/blob/master/convert_obj_to_mujoco_msh.py
'''

import struct
import os
import sys

def generate_msh_file(obj_f_name, msh_f_name=None, verbose=False):

    if obj_f_name[-4:] != ".obj":
        return
        
    if msh_f_name is None:
        msh_f_name = obj_f_name[:-4] + ".msh"

    v = []
    vt = []
    vn = []
    f = []

    obj = open(obj_f_name, 'r')
    for line in obj:
        arr = line.split()
        if len(arr) == 0:
            continue

        if arr[0] == 'v':
            assert len(arr)==4
            v.append([float(arr[i]) for i in range(1, 4)])
        elif arr[0] == 'vt':
            assert len(arr) == 3
            vt.append([float(arr[i]) for i in range(1, 3)])
        elif arr[0] == 'vn':
            assert len(arr) == 4
            vn.append([float(arr[i]) for i in range(1, 4)])
        elif arr[0] == 'f':
                tmp = [arr[i].split("/") for i in range(1,len(arr))]
                for i in range(1, len(tmp)-1):
                    f.append([tmp[0], tmp[i], tmp[i+1]])

    output_v = []
    output_vn = []
    output_vt = []
    output_f = []


    for i in f:
        assert len(i) == 3
        for j in i:
            v_idx = int(j[0])-1
            vt_idx = int(j[1]) - 1
            vn_idx = int(j[2]) - 1
            output_v.append(v[v_idx])
            output_vt.append(vt[vt_idx])
            output_vn.append(vn[vn_idx])
        output_f.append([len(output_v)-3, len(output_v)-2, len(output_v)-1])

    # print("len(v)=", len(output_v))
    # print("len(vn)=", len(output_vn))
    # print("len(vt)=", len(output_vt))
    # print("len(f)=", len(output_f))
    out = open(msh_f_name, 'wb')

    integer_packing = "=i"
    float_packing = "=f"

    def int_to_bytes(input):
        assert type(input) == int
        return struct.pack(integer_packing, input)

    def float_to_bytes(input):
        assert type(input) == float
        return struct.pack(float_packing, input)


    out.write(int_to_bytes(len(output_v)))
    out.write(int_to_bytes(len(output_vn)))
    out.write(int_to_bytes(len(output_vt)))
    out.write(int_to_bytes(len(output_f)))

    for i in output_v:
        for j in range(3):
            out.write(float_to_bytes(i[j]))
    for i in output_vn:
        for j in range(3):
            out.write(float_to_bytes(i[j]))
    for i in output_vt:
        out.write(float_to_bytes(i[0]))
        out.write(float_to_bytes(1-i[1]))
    for i in output_f:
        for j in range(3):
            out.write(int_to_bytes(i[j]))
    out.close()
    # print("actual file size=", os.path.getsize(msh_f_name))

    expected_file_size = 16 + 12*(len(output_v) + len(output_vn) + len(output_f)) + 8*len(output_vt)
    # print("expected_file_size=", expected_file_size)

    if verbose:
        print('done with... ' + obj_f_name)

if __name__ == '__main__':

    obj_name = sys.argv[1]
    # print(objs_dir_path)
    # obj_names = [i for i in os.listdir(objs_dir_path)]

    # for obj_name in obj_names:

    #     print(obj_name)
        
    #     obj_f_name = objs_dir_path + obj_name
    #     generate_msh_file(obj_f_name, verbose=True)

    obj_f_name = f'../test/test_objects/{obj_name}.obj'

    print(obj_f_name)

    generate_msh_file(obj_f_name, verbose=True)