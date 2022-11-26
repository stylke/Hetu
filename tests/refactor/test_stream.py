import hetu
import pickle
import unittest
import numpy as np

class TestStream(unittest.TestCase):

    _test_args = [
        ("cpu", 0), 
        ("cuda", 1), 
        ("cuda:1", 2), 
    ]

    def test_stream_getter(self):
        print(hetu.stream("cpu", np.int32(3)))
        print(hetu.stream("cpu", np.int64(4)))
        print(hetu.stream("cpu", np.int16(4)))
        print(hetu.stream("cpu", np.int8(4)))
        print(hetu.stream("cpu", np.uint8(4)))
        # self.assertEqual(hetu.stream("cuda:1", 0).device, hetu.device("cuda:1"))
        # self.assertEqual(hetu.stream("cpu", 0).device_type, "cpu")
        # self.assertEqual(hetu.stream("cuda", 0).device_type, "cuda")
        # self.assertEqual(hetu.stream("cuda", 0).device_index, 0)
        # self.assertEqual(hetu.stream("cuda:2", 0).device_index, 2)
        # self.assertEqual(hetu.stream("cuda", 0).stream_index, 0)
        # self.assertEqual(hetu.stream("cuda", 3).stream_index, 3)

    # def test_device_cmp(self):
    #     streams = [hetu.stream(*args) for args in TestStream._test_args]
    #     for i in range(len(TestStream._test_args) - 1):
    #         self.assertEqual(streams[i], hetu.stream(*TestStream._test_args[i]))
    #         with self.assertRaises(RuntimeError):
    #             self.assertLess(streams[i], streams[i + 1])

    # def test_stream_pickle(self):
    #     for args in TestStream._test_args:
    #         stream = hetu.stream(*args)
    #         stream_dumped = pickle.dumps(stream)
    #         stream_loaded = pickle.loads(stream_dumped)
    #         self.assertEqual(stream, stream_loaded)

if __name__ == "__main__":
    unittest.main()

