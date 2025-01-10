import grpc
import numpy as np
import json
import sys
from hetu.rpc import heturpc_pb2_grpc
from hetu.rpc import heturpc_pb2
from .const import *

def serialize_keys(data):
    if isinstance(data, dict):
        serialized_dict = {}
        for key, value in data.items():
            # 如果键是整数，转换为字符串并添加标记
            if isinstance(key, int):
                serialized_key = f"__int__{key}"
            elif isinstance(key, float):
                serialized_key = f"__float__{key}"
            else:
                serialized_key = key
            # 递归处理嵌套结构
            serialized_dict[serialized_key] = serialize_keys(value)
        return serialized_dict
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        return [serialize_keys(item) for item in data]
    else:
        # 其他类型直接返回
        return data
    
def deserialize_keys(data):
    if isinstance(data, dict):
        deserialized_dict = {}
        for key, value in data.items():
            # 如果键是带标记的字符串，还原为整数
            if isinstance(key, str) and key.startswith("__int__"):
                deserialized_key = int(key[len("__int__"):])
            elif isinstance(key, str) and key.startswith("__float__"):
                deserialized_key = int(key[len("__float__"):])
            else:
                deserialized_key = key
            # 递归处理嵌套结构
            deserialized_dict[deserialized_key] = deserialize_keys(value)
        return deserialized_dict
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        return [deserialize_keys(item) for item in data]
    else:
        # 其他类型直接返回
        return data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def numpy_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'ndarray':
        return np.array(obj['data'], dtype=obj['dtype'])
    return obj

class RemoteDict:
    def __init__(self, client, dict_name):
        self.client = client
        self.dict_name = dict_name

    def put(self, key, value):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        if isinstance(value, (int, float, dict, list, np.ndarray, np.number)):
            value = json.dumps(serialize_keys(value), cls=NumpyEncoder)
        elif not isinstance(value, str):
            sys.stderr.write("Value must be int, float, str, dict, list, or numpy type")
            raise ValueError("Value must be int, float, str, dict, list, or numpy type")
        self.client.stub.PutJson(heturpc_pb2.PutJsonRequest(key=full_key, value=value))

    def get(self, key):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        response = self.client.stub.GetJson(heturpc_pb2.GetJsonRequest(key=full_key))
        value = response.value
        try:
            value = deserialize_keys(json.loads(value, object_hook=numpy_decoder))
        except json.JSONDecodeError as e:
            sys.stderr.write(f"JSON decoding error, unable to parse value: {value}")
            raise e
        return value

    def remove(self, key):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        response = self.client.stub.RemoveJson(heturpc_pb2.RemoveJsonRequest(key=full_key))
        return response.message

    def get_many(self, keys):
        return {key: self.get(key) for key in keys}

class KeyValueStoreClient:
    def __init__(self, address='localhost:50051'):
        self.address = address
        self.channel = grpc.insecure_channel(self.address)
        self.stub = heturpc_pb2_grpc.DeviceControllerStub(self.channel)

    def register_dict(self, dict_name):
        return RemoteDict(self, dict_name)

# example
if __name__ == '__main__':
    client = KeyValueStoreClient(address='localhost:50051')

    # Register dictionaries
    data_store = client.register_dict('test_store')

    # Store nested structures containing numpy arrays
    print("Storing nested structure...")
    nested_data = {
        "user1": {
            "profile": {
                "name": "Alice",
                "scores": np.array([95, 88, 92])
            },
            "metrics": np.array([0.5, 0.8])
        },
        "user2": {
            "profile": {
                "name": "Bob",
                "scores": np.array([75, 85, 80])
            },
            "metrics": np.array([0.7, 0.6])
        }
    }
    data_store.put('nested_data', nested_data)

    # Retrieve nested structure
    print("Retrieving nested structure...")
    retrieved_data = data_store.get('nested_data')
    print(f"Retrieved data: {retrieved_data}")
    print(f"Is 'user1->metrics' a numpy array? {isinstance(retrieved_data['user1']['metrics'], np.ndarray)}")
