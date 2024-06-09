from concurrent import futures
import logging
import multiprocessing.spawn

import grpc
import heturpc_pb2
import heturpc_pb2_grpc

import threading
import time
import multiprocessing
import argparse

MAX_UNFOUND_TIMES = 10000


class DeviceController(heturpc_pb2_grpc.DeviceControllerServicer):
    def __init__(self, arr) -> None:
        super().__init__()
        self.arr = arr
        #locks
        self.lock = threading.Lock()
        self.double_lock = threading.Lock()
        self.int_lock = threading.Lock()
        self.string_lock = threading.Lock()
        self.bytes_lock = threading.Lock()
        self.json_lock = threading.Lock()
        # for start
        self.worldsize = 0
        self.local_worldsizes = {}
        self.local_ranks = {}
        self.nodenames = []
        self.hostnames = {}
        self.num_hostnames = 0
        self.deviceinfos = {}
        self.num_deviceinfos = 0
        self.nccl_ids = {}
        self.exit_nums = 0
        self.barrier_nums = {}
        self.barrier_end_nums = {}
        #dicts
        self.double_dict = {}
        self.int_dict = {}
        self.string_dict = {}
        self.bytes_dict = {}
        self.json_dict = {}

    def Connect(self, request, context):
        self.lock.acquire()
        nodename = "node-" + request.hostname
        if nodename not in self.local_worldsizes:
            self.local_worldsizes[nodename] = 0
            self.nodenames.append(nodename)
        self.local_worldsizes[nodename] = self.local_worldsizes[nodename] + 1
        self.worldsize += 1
        self.lock.release()
        return heturpc_pb2.ConnectReply(status=1)
        
    def GetRank(self, request, context):
        local_rank = 0
        self.lock.acquire()
        nodename = "node-" + request.name
        if nodename not in self.local_ranks:
            self.local_ranks[nodename] = 0
        for nodename_ in self.nodenames:
            if nodename_ == nodename:
                local_rank += self.local_ranks[nodename]
                self.local_ranks[nodename] = self.local_ranks[nodename] + 1
                break
            else:
                local_rank += self.local_worldsizes[nodename_]
        print(request.name, " ", local_rank)
        self.lock.release()
        return heturpc_pb2.RankReply(rank=local_rank)
    
    def CommitHostName(self, request, context):
        self.lock.acquire()
        self.num_hostnames += 1
        self.hostnames[request.rank] = request.hostname
        self.lock.release()
        return heturpc_pb2.CommitHostNameReply(status=1)

    def GetHostName(self, request, context):
        while(request.rank not in self.hostnames):
            time.sleep(0.0001)
        self.lock.acquire()
        hostname = self.hostnames[request.rank]
        self.lock.release()
        return heturpc_pb2.GetHostNameReply(hostname=hostname)

    def CommitDeviceInfo(self, request, context):
        self.lock.acquire()
        self.num_deviceinfos += 1
        self.deviceinfos[request.rank] = (request.type, request.index, request.multiplex)
        self.lock.release()
        return heturpc_pb2.CommitDeviceInfoReply(status=1)

    def GetDeviceInfo(self, request, context):
        while(request.rank not in self.deviceinfos):
            time.sleep(0.0001)
        self.lock.acquire()
        mtype, mindex, mmultiplex = self.deviceinfos[request.rank]
        self.lock.release()
        return heturpc_pb2.GetDeviceInfoReply(type=mtype, index=mindex, multiplex=mmultiplex)

    def CommitNcclId(self, request, context):
        self.lock.acquire()
        world_rank = tuple(request.world_rank) + (int(request.stream_id),)
        # print("RECEIVE:", world_rank, type(world_rank))
        self.nccl_ids[world_rank] = request.nccl_id
        self.lock.release()
        return heturpc_pb2.CommitNcclIdReply(status=1)

    def GetNcclId(self, request, context):
        world_rank = tuple(request.world_rank) + (int(request.stream_id),)
        while(world_rank not in self.nccl_ids):
            time.sleep(0.0001)
        self.lock.acquire()
        nccl_id = self.nccl_ids[world_rank]
        self.lock.release()
        return heturpc_pb2.GetNcclIdReply(nccl_id=nccl_id)

    def Exit(self, request, context):
        self.lock.acquire()
        self.exit_nums += 1
        self.lock.release()
        print(self.exit_nums, " ", self.worldsize)
        while(self.exit_nums != self.worldsize):
            time.sleep(0.0001)
        self.arr[0] = 1
        return heturpc_pb2.ExitReply(status=1)

    def PutDouble(self, request, context):
        self.double_lock.acquire()
        self.double_dict[request.key] = request.value
        self.double_lock.release()
        return heturpc_pb2.PutDoubleReply(status=1)

    def GetDouble(self, request, context):
        while(request.key not in self.int_dict):
            time.sleep(0.0001)
        self.int_lock.acquire()
        value = self.int_dict[request.key]
        self.int_lock.release()
        return heturpc_pb2.GetDoubleReply(value=value)

    def RemoveDouble(self, request, context):
        unfound_times = 0
        while(request.key not in self.double_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveDoubleReply(message="not found:" + request.key)
        else:
            self.double_dict.pop(request.key)
        return heturpc_pb2.RemoveDoubleReply(message="already remove:" + request.key)
    
    def PutInt(self, request, context):
        self.int_lock.acquire()
        self.int_dict[request.key] = request.value
        self.int_lock.release()
        return heturpc_pb2.PutIntReply(status=1)

    def GetInt(self, request, context):
        while(request.key not in self.double_dict):
            time.sleep(0.0001)
        self.double_lock.acquire()
        value = self.double_dict[request.key]
        self.double_lock.release()
        return heturpc_pb2.GetIntReply(value=value)

    def RemoveInt(self, request, context):
        unfound_times = 0
        while(request.key not in self.int_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveIntReply(message="not found:" + request.key)
        else:
            self.int_dict.pop(request.key)
        return heturpc_pb2.RemoveIntReply(message="already remove:" + request.key)
    
    def PutString(self, request, context):
        self.string_lock.acquire()
        self.string_dict[request.key] = request.value
        self.string_lock.release()
        return heturpc_pb2.PutStringReply(status=1)

    def GetString(self, request, context):
        while(request.key not in self.string_dict):
            time.sleep(0.0001)
        self.string_lock.acquire()
        value = self.string_dict[request.key]
        self.string_lock.release()
        return heturpc_pb2.GetStringReply(value=value)

    def RemoveString(self, request, context):
        unfound_times = 0
        while(request.key not in self.string_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveStringReply(message="not found:" + request.key)
        else:
            self.string_dict.pop(request.key)
        return heturpc_pb2.RemoveStringReply(message="already remove:" + request.key)
    
    def PutBytes(self, request, context):
        self.bytes_lock.acquire()
        self.bytes_dict[request.key] = request.value
        self.bytes_lock.release()
        return heturpc_pb2.PutBytesReply(status=1)

    def GetBytes(self, request, context):
        while(request.key not in self.bytes_dict):
            time.sleep(0.0001)
        self.bytes_lock.acquire()
        value = self.bytes_dict[request.key]
        self.bytes_lock.release()
        return heturpc_pb2.GetBytesReply(value=value)
    
    def RemoveBytes(self, request, context):
        unfound_times = 0
        while(request.key not in self.bytes_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveBytesReply(message="not found:" + request.key)
        else:
            self.bytes_dict.pop(request.key)
        return heturpc_pb2.RemoveBytesReply(message="already remove:" + request.key)

    def PutJson(self, request, context):
        self.json_lock.acquire()
        self.json_dict[request.key] = request.value
        import json
        my_json = json.loads(request.value)
        # print("ReCEIVEJSON:", my_json)
        self.json_lock.release()
        return heturpc_pb2.PutJsonReply(status=1)

    def GetJson(self, request, context):
        while(request.key not in self.json_dict):
            time.sleep(0.0001)
        self.json_lock.acquire()
        value = self.json_dict[request.key]
        self.json_lock.release()
        return heturpc_pb2.GetJsonReply(value=value)
    
    def RemoveJson(self, request, context):
        unfound_times = 0
        while(request.key not in self.json_dict and unfound_times < MAX_UNFOUND_TIMES):
            time.sleep(0.0001)
            unfound_times += 1
        if unfound_times == MAX_UNFOUND_TIMES:
            return heturpc_pb2.RemoveJsonReply(message="not found:" + request.key)
        else:
            self.json_dict.pop(request.key)
        return heturpc_pb2.RemoveJsonReply(message="already remove:" + request.key)
    
    def Barrier(self, request, context):
        world_rank = tuple(request.world_rank)
        self.lock.acquire()
        if (world_rank not in self.barrier_nums):
            self.barrier_nums[world_rank] = 0
        self.lock.release()
        self.lock.acquire()
        if (request.rank == world_rank[0]):
            self.barrier_end_nums[world_rank] = 0
        self.barrier_nums[world_rank] += 1
        self.lock.release()
        # print("S1 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        while(self.barrier_nums[world_rank] < len(world_rank)):
            time.sleep(0.0001)
        # print("S2 Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        self.lock.acquire()
        self.barrier_end_nums[world_rank] += 1
        self.lock.release()
        while(self.barrier_end_nums[world_rank] < len(world_rank)):
            time.sleep(0.0001)
        if (request.rank == world_rank[0]):
            self.barrier_nums[world_rank] = 0
        # print("End Barrier:", self.barrier_nums[world_rank], " ", len(world_rank))
        return heturpc_pb2.BarrierReply(status=1)
    
    def SayHello(self, request, context):
        return heturpc_pb2.HetuReply(message="Hello, %s!" % request.name)

def serve(arr, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    heturpc_pb2_grpc.add_DeviceControllerServicer_to_server(DeviceController(arr), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=str, default='23457', help="server's port"
    )
    server_args = parser.parse_args()
    logging.basicConfig()
    arr = multiprocessing.Array("i",[0])
    p = multiprocessing.Process(target=serve, args=(arr, server_args.port))
    p.start()
    while (arr[0] == 0):
        time.sleep(1)
    print("Server Stopped.")
    p.terminate()