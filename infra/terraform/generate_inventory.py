#!/usr/bin/env python3

import json
import argparse
import subprocess

def run(command):
    return subprocess.run(command, capture_output=True, encoding='UTF-8')

def generate_inventory():
    host_vars = {}

    # get host node IP
    command = "terraform output --json host_vm_ips".split()
    host_ip = json.loads(run(command).stdout).pop()
    host_vars[host_ip] = { "ip": [host_ip] }

    # get worker node IPs
    command = "terraform output --json worker_vm_ips".split()
    worker_ips = json.loads(run(command).stdout)

    for ip in worker_ips:
        host_vars[ip] = { "ip": [ip] }

    # structure inventory
    inventory = {
        "_meta": { "hostvars": host_vars },
        "all": { "children": ["hostnode", "workers"] },
        "hostnode": { "hosts": [host_ip] },
        "workers": { "hosts": worker_ips }
    }

    return json.dumps(inventory, indent=4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate dynamic inventory from Terraform")
    mo = ap.add_mutually_exclusive_group()
    mo.add_argument("--list", action="store", nargs="*", default="dummy")
    mo.add_argument("--host", action="store")

    args = ap.parse_args()

    if args.host:
        print(json.dumps({}))
    else:
        print(generate_inventory())