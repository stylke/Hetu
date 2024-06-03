import os
import argparse
import yaml
from pssh.clients import ParallelSSHClient
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def pssh(args):
    hostnames = []
    if args.hosts is None:
        hostnames = ['localhost'] * args.ngpus
    else:
        host_info = read_yaml(args.hosts)
        for host in host_info['hosts']:
            hostnames.append(host['addr'])
    print("HostNames:", hostnames)
    client = ParallelSSHClient(hostnames)
    train_command = args.command
    print(train_command)
    cwd = os.getcwd()
    print(cwd)
    cmd1 = "cd " + cwd
    conda_env = os.environ["CONDA_PREFIX"]
    # output0 =client.run_command(cmd1)
    cmd2 = cmd1 + " && source activate && conda activate " + conda_env \
                + " && source ../../../hetu_refactor.exp && " + train_command
    print(cmd2)
    output = client.run_command(cmd2)
    for host_out in output:
        for line in host_out.stderr:
            print(line)
        for line in host_out.stdout:
            print(line)
        exit_code = host_out.exit_code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command", type=str, default='uname', help="command for pssh"
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='50051', help="server's port"
    )
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num gpus"
    )
    parser.add_argument(
        "--hosts", type=str, help="server's port"
    )
    args = parser.parse_args()
    os.system("python ../../../python_refactor/hetu/rpc/heturpc_server.py --port " + args.server_port + "&")
    pssh(args)