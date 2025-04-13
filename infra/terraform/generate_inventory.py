#!/usr/bin/env python3

import json
import subprocess
import argparse

def run(command):
    result = subprocess.run(command, capture_output=True, encoding='UTF-8')
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(result.stderr)
        return None
    return result.stdout

def generate_inventory():
    # retrieve management node ip (hostnode)
    command = "terraform output --json host_vm_ips".split()
    mgmt_output = run(command)
    if not mgmt_output:
        raise ValueError("Failed to retrieve host_vm_ips from Terraform output.")

    ip_data = json.loads(mgmt_output)
    # access the first ip in the list directly
    hostnode = ip_data[0]  

    # initialize host_vars
    host_vars = {}
    host_vars[hostnode] = {"ip": [hostnode]}

    # retrieve worker node ips
    workers = []
    command = "terraform output --json worker_vm_ips".split()
    worker_output = run(command)
    if not worker_output:
        raise ValueError("Failed to retrieve worker_vm_ips from Terraform output.")

    ip_data = json.loads(worker_output)

    # directly iterate over the list of worker ips
    for worker_ip in ip_data:  
        name = worker_ip
        host_vars[name] = {"ip": [name]}
        workers.append(name)

    # define the inventory structure
    _meta = {"hostvars": host_vars}
    _all = {"children": ["hostnode", "workers"]}

    _workers = {"hosts": workers}
    _hostnode = {"hosts" : [hostnode]}

    # create the final json structure for the inventory
    _jd = {
        "_meta": _meta,
        "all": _all,
        "workers": _workers,
        "hostnode": _hostnode
    }

    return json.dumps(_jd, indent=4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate a cluster inventory from Terraform.",
        prog=__file__
    )

    mo = ap.add_mutually_exclusive_group()
    mo.add_argument("--list", action="store", nargs="*", default="dummy", help="Show JSON of all managed hosts")
    mo.add_argument("--host", action="store", help="Display vars related to the host")

    args = ap.parse_args()

    if args.host:
        print(json.dumps({}))
    elif len(args.list) >= 0:
        jd = generate_inventory()
        print(jd)
    else:
        raise ValueError("Expecting either --host $HOSTNAME or --list")