import json

type_list = []
with open('comm_logs_rank2.json', 'r') as f:
    data_rank0 = json.load(f)
    for i in range(len(data_rank0)):
        if data_rank0[i]['op_type'] not in type_list:
            type_list.append(data_rank0[i]['op_type'])
print(type_list)