# -*- coding: utf-8 -*-
# vim:expandtab:autoindent:tabstop=4:shiftwidth=4:filetype=python:textwidth=0:
# License: GPL2 or later see COPYING
# Copyright (C) 2025

"""
NAT-based network isolation for Mock build environments.

Creates a veth pair between the host and an isolated network namespace,
configures iptables/ip6tables NAT rules, and tears everything down
cleanly after the build command completes.
"""

import ctypes
import fcntl
import ipaddress
import os
import shutil
import subprocess
import tempfile
import uuid

from pyroute2 import IPRoute
from pyroute2 import netns

from .trace_decorator import getLog, traceLog

# setns() for entering a pre-created namespace in child processes
_libc = ctypes.CDLL(None, use_errno=True)
_libc.setns.argtypes = [ctypes.c_int, ctypes.c_int]
_libc.setns.restype = ctypes.c_int

CLONE_NEWNET = 0x40000000

# Lock file for subnet pool allocation
_LOCK_DIR = "/run/mock-nat"
_LOCK_FILE = os.path.join(_LOCK_DIR, "subnet.lock")

# Default fallback DNS servers used when the host's resolv.conf only contains
# loopback addresses (e.g. systemd-resolved 127.0.0.53) which are
# unreachable from inside the NAT namespace.
_DEFAULT_FALLBACK_DNS = ['8.8.8.8', '1.1.1.1']


def _read_host_nameservers(fallback_dns=None):
    """
    Read nameserver addresses from the host's /etc/resolv.conf.

    Returns a list of nameserver IP strings, filtering out loopback
    addresses (127.x.x.x, ::1) that are unreachable from a NAT namespace.
    If no non-loopback nameservers are found, returns fallback_dns
    (or the built-in default if not specified).
    """
    if fallback_dns is None:
        fallback_dns = _DEFAULT_FALLBACK_DNS

    nameservers = []
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('nameserver'):
                    parts = line.split()
                    if len(parts) >= 2:
                        ns = parts[1]
                        # Filter out loopback addresses — they refer to the
                        # host's own resolver stub (e.g. systemd-resolved)
                        # which is not reachable from the NAT namespace.
                        if not ns.startswith('127.') and ns != '::1':
                            nameservers.append(ns)
    except IOError:
        pass

    if not nameservers:
        nameservers = list(fallback_dns)
        getLog().info("NAT: no non-loopback nameservers found, using fallback: %s",
                      nameservers)
    return nameservers


def _ensure_lock_dir():
    """Create the lock directory if it doesn't exist."""
    try:
        os.makedirs(_LOCK_DIR, mode=0o755, exist_ok=True)
    except OSError:
        pass


def _subdivide(cidr_str, prefixlen):
    """
    Subdivide a CIDR block into subnets of the given prefix length.

    Args:
        cidr_str: e.g. '192.168.200.0/24' or 'fd00:dead::/48'
        prefixlen: e.g. 29 (IPv4) or 64 (IPv6)

    Returns:
        list of ipaddress.IPv4Network or IPv6Network objects
    """
    if not cidr_str:
        return []
    try:
        net = ipaddress.ip_network(cidr_str, strict=False)
    except ValueError as e:
        raise ValueError("Invalid network_veth_subnet_block value %r: %s" % (cidr_str, e))
    if net.prefixlen >= prefixlen:
        raise ValueError(
            "network_veth_subnet_block %r prefix /%d is not larger than "
            "the subdivision prefix /%d" % (cidr_str, net.prefixlen, prefixlen))
    return list(net.subnets(new_prefix=prefixlen))


class SubnetPool:
    """
    Thread-safe and process-safe allocator for veth subnets.

    Uses file locking to prevent concurrent builds from receiving
    the same subnet.

    Accepts either a CIDR block (which is automatically subdivided into
    /30 slices for IPv4 or /64 slices for IPv6) or an explicit list of
    subnets (legacy format kept for backwards compatibility).
    """

    # Subdivision prefix lengths.
    # /30 gives exactly 4 addresses: network, host (.1), container (.2),
    # broadcast — the minimum needed for a point-to-point veth pair.
    _V4_PREFIX = 30
    _V6_PREFIX = 64

    def __init__(self, pool_v4, pool_v6=None):
        """
        Args:
            pool_v4: A CIDR block string (e.g. '192.168.200.0/24') that
                will be subdivided into /30 subnets, OR a list of explicit
                /30 subnet strings for backwards compatibility.
            pool_v6: A CIDR block string (e.g. 'fd00:dead:beef::/48') that
                will be subdivided into /64 subnets, OR a list of explicit
                subnet strings, OR None to disable IPv6.
        """
        if isinstance(pool_v4, str):
            self.pool_v4 = _subdivide(pool_v4, self._V4_PREFIX)
        else:
            self.pool_v4 = [ipaddress.IPv4Network(n) for n in (pool_v4 or [])]

        if isinstance(pool_v6, str):
            self.pool_v6 = _subdivide(pool_v6, self._V6_PREFIX)
        else:
            self.pool_v6 = [ipaddress.IPv6Network(n) for n in (pool_v6 or [])]

    def allocate(self):
        """
        Allocate a unique subnet pair (v4, v6) from the pool.

        Returns:
            (ipv4_network, ipv6_network_or_None)
        """
        _ensure_lock_dir()
        lock_fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            # Read current allocation state using raw fd I/O
            os.lseek(lock_fd, 0, os.SEEK_SET)
            data = os.read(lock_fd, 65536).decode('utf-8')
            allocated = set()
            for line in data.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    allocated.add(line)

            # Find an unallocated v4 subnet
            for net in self.pool_v4:
                key = str(net)
                if key not in allocated:
                    # Find matching v6 subnet (same index)
                    v6_net = None
                    idx = self.pool_v4.index(net)
                    if idx < len(self.pool_v6):
                        v6_net = self.pool_v6[idx]
                        v6_key = str(v6_net)
                        if v6_key in allocated:
                            # v6 already taken, skip this pair
                            continue
                        allocated.add(v6_key)

                    allocated.add(key)
                    # Write back using raw fd I/O
                    os.ftruncate(lock_fd, 0)
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    os.write(lock_fd, '\n'.join(sorted(allocated)).encode('utf-8') + b'\n')
                    return net, v6_net

            raise RuntimeError("No available subnets in NAT pool. "
                               "Increase network_veth_subnet_block size.")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def release(self, v4_net, v6_net=None):
        """Release a previously allocated subnet pair."""
        _ensure_lock_dir()
        lock_fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            os.lseek(lock_fd, 0, os.SEEK_SET)
            data = os.read(lock_fd, 65536).decode('utf-8')
            allocated = set()
            for line in data.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    allocated.add(line)

            allocated.discard(str(v4_net))
            if v6_net:
                allocated.discard(str(v6_net))

            # Write back using raw fd I/O
            os.ftruncate(lock_fd, 0)
            os.lseek(lock_fd, 0, os.SEEK_SET)
            content = '\n'.join(sorted(allocated))
            if content:
                content += '\n'
            os.write(lock_fd, content.encode('utf-8'))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


class NatNetwork:
    """
    Manages a veth pair + NAT for a single build command.

    Usage:
        net = NatNetwork(config)
        ns_path = net.setup()         # parent process, before fork
        # ... fork child, pass ns_path to ChildPreExec ...
        net.teardown()                # parent process, after child exits
    """

    def __init__(self, config, chroot_path=None):
        """
        Args:
            config: Mock config_opts dict containing:
                network_veth_subnet_block: IPv4 CIDR block (auto-subdivided
                    into /29 subnets), e.g. '192.168.200.0/24'
                network_veth_subnet_block_v6: IPv6 CIDR block (auto-subdivided
                    into /64 subnets), e.g. 'fd00:dead:beef::/48' (optional)
                network_fallback_dns: list of DNS server IPs used when the
                    host resolv.conf only has loopback entries (optional)
            chroot_path: Path to the chroot root directory, used for
                writing DNS configuration (resolv.conf) when the NAT
                namespace needs network access.
        """
        self.config = config
        self.chroot_path = chroot_path
        self.ns_name = "m-" + uuid.uuid4().hex[:8]
        self.host_if = "vm-" + self.ns_name
        self.cont_if = "host0"
        self.ns_path = "/var/run/netns/" + self.ns_name

        self.v4_net = None
        self.v6_net = None
        self.host_ip_v4 = None
        self.cont_ip_v4 = None
        self.host_ip_v6 = None
        self.cont_ip_v6 = None
        self.mask_v4 = None
        self.mask_v6 = None

        # For DNS resolv.conf save/restore
        self._resolv_backup = None

        self._pool = SubnetPool(
            config.get('network_veth_subnet_block', ''),
            config.get('network_veth_subnet_block_v6', ''),
        )

    @traceLog()
    def setup(self):
        """
        Create and configure the NAT network. Called in the PARENT process
        before forking the child.

        Returns:
            ns_path (str): Path to the network namespace, e.g.
                           /var/run/netns/mock-a1b2c3d4
        """
        log = getLog()
        self.v4_net, self.v6_net = self._pool.allocate()

        # Derive host/container IPs from the subnet
        # For a /29, we use .1 for host and .2 for container
        hosts_v4 = list(self.v4_net.hosts())
        self.host_ip_v4 = str(hosts_v4[0])
        self.cont_ip_v4 = str(hosts_v4[1])
        self.mask_v4 = self.v4_net.prefixlen

        if self.v6_net:
            # For IPv6, use ::1 for host and ::2 for container
            self.host_ip_v6 = str(self.v6_net.network_address + 1)
            self.cont_ip_v6 = str(self.v6_net.network_address + 2)
            self.mask_v6 = self.v6_net.prefixlen

        log.info("NAT network: namespace=%s, v4=%s, host_if=%s",
                 self.ns_name, self.v4_net, self.host_if)

        # 1. Create network namespace
        netns.create(self.ns_name)

        # 2. Create veth pair in host namespace
        ipr = IPRoute()
        ipr.link('add', ifname=self.host_if, kind='veth', peer=self.cont_if)

        # 3. Move container end into the new namespace
        cont_idx = ipr.link_lookup(ifname=self.cont_if)[0]
        ipr.link('set', index=cont_idx, net_ns_fd=self.ns_name)

        # 4. Configure host end
        host_idx = ipr.link_lookup(ifname=self.host_if)[0]
        ipr.link('set', index=host_idx, state='up')
        ipr.addr('add', index=host_idx, address=self.host_ip_v4,
                 mask=self.mask_v4)

        # 5. Configure IPv6 on host end
        if self.host_ip_v6:
            ipr.addr('add', index=host_idx, address=self.host_ip_v6,
                     mask=self.mask_v6)

        # 6. Set up iptables NAT for IPv4
        self._setup_iptables_nat()

        # 7. Set up ip6tables NAT for IPv6
        if self.v6_net:
            self._setup_ip6tables_nat()

        # 8. Enable IP forwarding (idempotent)
        self._enable_forwarding()

        # 9. Set up DNS resolution in the chroot
        if self.chroot_path:
            self._setup_dns()

        return self.ns_path

    @traceLog()
    def _setup_iptables_nat(self):
        """Add iptables MASQUERADE rule for the v4 subnet."""
        subprocess.run(
            ['iptables', '-t', 'nat', '-I', 'POSTROUTING',
             '-s', str(self.v4_net),
             '-j', 'MASQUERADE',
             '-m', 'comment', '--comment', 'mock-' + self.ns_name],
            check=True,
        )
        # Allow forwarding from the veth subnet
        subprocess.run(
            ['iptables', '-I', 'FORWARD',
             '-i', self.host_if,
             '-j', 'ACCEPT'],
            check=True,
        )
        subprocess.run(
            ['iptables', '-I', 'FORWARD',
             '-o', self.host_if,
             '-j', 'ACCEPT'],
            check=True,
        )

    @traceLog()
    def _setup_ip6tables_nat(self):
        """Add ip6tables MASQUERADE rule for the v6 subnet."""
        subprocess.run(
            ['ip6tables', '-t', 'nat', '-I', 'POSTROUTING',
             '-s', str(self.v6_net),
             '-j', 'MASQUERADE',
             '-m', 'comment', '--comment', 'mock-' + self.ns_name],
            check=True,
        )
        subprocess.run(
            ['ip6tables', '-I', 'FORWARD',
             '-i', self.host_if,
             '-j', 'ACCEPT'],
            check=True,
        )
        subprocess.run(
            ['ip6tables', '-I', 'FORWARD',
             '-o', self.host_if,
             '-j', 'ACCEPT'],
            check=True,
        )

    @traceLog()
    def _enable_forwarding(self):
        """Enable IPv4 and IPv6 forwarding (idempotent)."""
        subprocess.run(['sysctl', '-w', 'net.ipv4.ip_forward=1'],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['sysctl', '-w', 'net.ipv6.conf.all.forwarding=1'],
                       check=True, stdout=subprocess.DEVNULL)

    @traceLog()
    def configure_container_ns(self):
        """
        Configure the container-side network interface.

        Called INSIDE the child process, after entering the network namespace
        (via setns), before exec'ing nspawn.
        """
        ipr = IPRoute()

        # Bring up loopback
        lo_idx = ipr.link_lookup(ifname='lo')[0]
        ipr.link('set', index=lo_idx, state='up')

        # Configure container-side veth
        cont_idx = ipr.link_lookup(ifname=self.cont_if)
        if not cont_idx:
            # Interface not found — this can happen if the namespace
            # setup failed silently
            getLog().warning("NAT: container interface %s not found in namespace",
                             self.cont_if)
            return

        cont_idx = cont_idx[0]
        ipr.link('set', index=cont_idx, state='up')

        # IPv4 address and default route
        ipr.addr('add', index=cont_idx, address=self.cont_ip_v4,
                 mask=self.mask_v4)
        ipr.route('add', dst='default', gateway=self.host_ip_v4)

        # IPv6 address and default route
        if self.cont_ip_v6:
            ipr.addr('add', index=cont_idx, address=self.cont_ip_v6,
                     mask=self.mask_v6)
            ipr.route('add', dst='default', gateway=self.host_ip_v6)

    @traceLog()
    def teardown(self):
        """
        Tear down the NAT network. Called in the PARENT process after
        the child exits.
        """
        log = getLog()

        # Restore the original resolv.conf
        if self.chroot_path:
            self._teardown_dns()

        # Remove iptables rules by comment marker
        self._remove_iptables_rules()

        # Remove ip6tables rules
        if self.v6_net:
            self._remove_ip6tables_rules()

        # Delete the veth pair (deleting the host end removes both ends)
        try:
            ipr = IPRoute()
            host_idx = ipr.link_lookup(ifname=self.host_if)
            if host_idx:
                ipr.link('del', index=host_idx[0])
        except Exception as e:
            log.warning("NAT: failed to delete veth %s: %s", self.host_if, e)

        # Delete the network namespace
        try:
            netns.remove(self.ns_name)
        except Exception as e:
            log.warning("NAT: failed to remove namespace %s: %s", self.ns_name, e)

        # Release the subnet back to the pool
        self._pool.release(self.v4_net, self.v6_net)

        log.info("NAT network torn down: namespace=%s", self.ns_name)

    @traceLog()
    def _setup_dns(self):
        """
        Write a working resolv.conf into the chroot for NAT namespace DNS.

        When the build environment is behind NAT, it can reach the same DNS
        servers the host uses (via NAT routing).  However, the chroot's
        resolv.conf may be empty (when rpmbuild_networking=False).  We
        back up the original and write one with usable nameservers.

        Loopback nameservers (127.x.x.x, ::1) from the host are filtered
        out because they are unreachable from the NAT namespace.  If no
        non-loopback nameservers are found, public fallback servers are
        used instead.
        """
        log = getLog()
        resolv_path = os.path.join(self.chroot_path, 'etc', 'resolv.conf')

        # Back up the existing resolv.conf
        try:
            with open(resolv_path, 'rb') as f:
                self._resolv_backup = f.read()
        except (IOError, OSError):
            self._resolv_backup = None

        # Build a new resolv.conf with reachable nameservers
        fallback_dns = self.config.get('network_fallback_dns', None)
        nameservers = _read_host_nameservers(fallback_dns=fallback_dns)
        lines = ['# Generated by mock NAT network isolation']
        for ns in nameservers:
            lines.append('nameserver {}'.format(ns))

        try:
            os.makedirs(os.path.dirname(resolv_path), exist_ok=True)
            with open(resolv_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            log.debug("NAT: wrote resolv.conf with nameservers: %s", nameservers)
        except (IOError, OSError) as e:
            log.warning("NAT: failed to write resolv.conf: %s", e)

    @traceLog()
    def _teardown_dns(self):
        """
        Restore the original resolv.conf that was backed up during _setup_dns.
        """
        log = getLog()
        if self._resolv_backup is None:
            return

        resolv_path = os.path.join(self.chroot_path, 'etc', 'resolv.conf')
        try:
            with open(resolv_path, 'wb') as f:
                f.write(self._resolv_backup)
            log.debug("NAT: restored original resolv.conf")
        except (IOError, OSError) as e:
            log.warning("NAT: failed to restore resolv.conf: %s", e)
        finally:
            self._resolv_backup = None

    @traceLog()
    def _remove_iptables_rules(self):
        """Remove iptables rules matching our comment marker."""
        marker = 'mock-' + self.ns_name
        try:
            # Delete NAT rule
            subprocess.run(
                ['iptables', '-t', 'nat', '-D', 'POSTROUTING',
                 '-s', str(self.v4_net),
                 '-j', 'MASQUERADE',
                 '-m', 'comment', '--comment', marker],
                check=False,  # don't fail if rule doesn't exist
            )
            # Delete FORWARD rules
            subprocess.run(
                ['iptables', '-D', 'FORWARD',
                 '-i', self.host_if,
                 '-j', 'ACCEPT'],
                check=False,
            )
            subprocess.run(
                ['iptables', '-D', 'FORWARD',
                 '-o', self.host_if,
                 '-j', 'ACCEPT'],
                check=False,
            )
        except Exception as e:
            getLog().warning("NAT: iptables cleanup failed: %s", e)

    @traceLog()
    def _remove_ip6tables_rules(self):
        """Remove ip6tables rules matching our comment marker."""
        marker = 'mock-' + self.ns_name
        try:
            subprocess.run(
                ['ip6tables', '-t', 'nat', '-D', 'POSTROUTING',
                 '-s', str(self.v6_net),
                 '-j', 'MASQUERADE',
                 '-m', 'comment', '--comment', marker],
                check=False,
            )
            subprocess.run(
                ['ip6tables', '-D', 'FORWARD',
                 '-i', self.host_if,
                 '-j', 'ACCEPT'],
                check=False,
            )
            subprocess.run(
                ['ip6tables', '-D', 'FORWARD',
                 '-o', self.host_if,
                 '-j', 'ACCEPT'],
                check=False,
            )
        except Exception as e:
            getLog().warning("NAT: ip6tables cleanup failed: %s", e)


def setns(ns_path, netns_type=CLONE_NEWNET):
    """
    Enter an existing network namespace. Used by the child process
    before exec'ing nspawn.

    Args:
        ns_path: Path to the network namespace, e.g. /var/run/netns/mock-xxx
        netns_type: Namespace type flag (default: CLONE_NEWNET)
    """
    log = getLog()
    try:
        fd = os.open(ns_path, os.O_RDONLY)
        ret = _libc.setns(fd, netns_type)
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        os.close(fd)
        log.debug("setns: entered namespace %s", ns_path)
    except Exception as e:
        log.error("setns: failed to enter namespace %s: %s", ns_path, e)
        raise
