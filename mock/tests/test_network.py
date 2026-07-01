"""Unit tests for mockbuild.network subnet allocation helpers."""

import ipaddress
import os
import fcntl
import subprocess

import pytest

from mockbuild import network


@pytest.fixture(autouse=True)
def isolate_lock_dir(tmp_path, monkeypatch):
    """Use a per-test temporary directory for the subnet lock file."""
    lock_dir = tmp_path / "nat-lock"
    lock_dir.mkdir()
    monkeypatch.setattr(network, "_LOCK_DIR", str(lock_dir))
    monkeypatch.setattr(network, "_LOCK_FILE", str(lock_dir / "subnet.lock"))


def _write_lock_file_from_entries(pool, entries):
    """Seed the lock file with the given entries, taking a real lock."""
    network._ensure_lock_dir()
    lock_fd = os.open(network._LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pool._write_lock_file(lock_fd, entries)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


class TestSubdivide:
    """Tests for the _subdivide helper."""

    def test_ipv4_subdivision(self):
        subnets = network._subdivide("192.168.200.0/24", 30)
        assert len(subnets) == 64
        assert subnets[0] == ipaddress.ip_network("192.168.200.0/30")
        assert subnets[-1] == ipaddress.ip_network("192.168.200.252/30")

    def test_ipv6_subdivision(self):
        subnets = network._subdivide("fd00:dead:beef::/56", 64)
        assert len(subnets) == 256
        assert subnets[0] == ipaddress.ip_network("fd00:dead:beef::/64")
        assert subnets[-1] == ipaddress.ip_network("fd00:dead:beef:ff::/64")

    def test_empty_string_returns_empty_list(self):
        assert network._subdivide("", 30) == []

    def test_invalid_cidr_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid network_veth_subnet_block"):
            network._subdivide("not-a-network", 30)

    def test_prefix_too_large_raises_value_error(self):
        with pytest.raises(ValueError, match="prefix /30 is not larger than"):
            network._subdivide("192.168.200.0/30", 30)


class TestSubnetPoolInit:
    """Tests for SubnetPool construction."""

    def test_ipv4_only(self):
        pool = network.SubnetPool("192.168.200.0/24")
        assert len(pool.pool_v4) == 64
        assert pool.pool_v6 == []

    def test_ipv4_and_ipv6(self):
        pool = network.SubnetPool("192.168.200.0/24", "fd00:dead:beef::/48")
        assert len(pool.pool_v4) == 64
        assert len(pool.pool_v6) == 65536

    def test_invalid_pool_raises(self):
        with pytest.raises(ValueError):
            network.SubnetPool("bad-cidr")


class TestParseLine:
    """Tests for SubnetPool._parse_line."""

    def test_full_line(self):
        subnet, pid, ns_name = network.SubnetPool._parse_line(
            "192.168.200.0/30 pid=12345 ns=m-a1b2c3d4"
        )
        assert subnet == "192.168.200.0/30"
        assert pid == 12345
        assert ns_name == "m-a1b2c3d4"

    def test_missing_ns(self):
        subnet, pid, ns_name = network.SubnetPool._parse_line(
            "192.168.200.0/30 pid=12345"
        )
        assert subnet == "192.168.200.0/30"
        assert pid == 12345
        assert ns_name is None

    def test_missing_pid(self):
        subnet, pid, ns_name = network.SubnetPool._parse_line(
            "192.168.200.0/30 ns=m-abc"
        )
        assert subnet == "192.168.200.0/30"
        assert pid is None
        assert ns_name == "m-abc"

    def test_malformed_pid_ignored(self):
        subnet, pid, ns_name = network.SubnetPool._parse_line(
            "192.168.200.0/30 pid=abc ns=m-abc"
        )
        assert subnet == "192.168.200.0/30"
        assert pid is None
        assert ns_name == "m-abc"

    def test_empty_line(self):
        subnet, pid, ns_name = network.SubnetPool._parse_line("")
        assert subnet == ""
        assert pid is None
        assert ns_name is None


class TestIsPidAlive:
    """Tests for SubnetPool._is_pid_alive."""

    def test_current_process_is_alive(self):
        assert network.SubnetPool._is_pid_alive(os.getpid()) is True

    def test_dead_process(self):
        proc = subprocess.Popen(
            ["python3", "-c", "exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert network.SubnetPool._is_pid_alive(proc.pid) is False

    def test_permission_error_treated_as_alive(self, monkeypatch):
        def raise_permission_error(*_args, **_kwargs):
            raise PermissionError("permission denied")

        monkeypatch.setattr(os, "kill", raise_permission_error)
        assert network.SubnetPool._is_pid_alive(1) is True


class TestAllocateRelease:
    """Tests for SubnetPool.allocate and SubnetPool.release."""

    def test_allocate_ipv4_only(self):
        pool = network.SubnetPool("192.168.200.0/29")
        v4_net, v6_net = pool.allocate()
        assert v4_net == ipaddress.ip_network("192.168.200.0/30")
        assert v6_net is None

    def test_allocate_ipv4_ipv6_pair(self):
        pool = network.SubnetPool("192.168.200.0/29", "fd00:dead:beef::/48")
        v4_net, v6_net = pool.allocate()
        assert v4_net == ipaddress.ip_network("192.168.200.0/30")
        assert v6_net == ipaddress.ip_network("fd00:dead:beef::/64")

    def test_allocate_writes_lock_file(self):
        pool = network.SubnetPool("192.168.200.0/29")
        v4_net, _ = pool.allocate(ns_name="m-test")
        content = open(network._LOCK_FILE).read()
        assert "192.168.200.0/30" in content
        assert f"pid={os.getpid()}" in content
        assert "ns=m-test" in content

    def test_release_removes_lock_entry(self):
        pool = network.SubnetPool("192.168.200.0/29")
        v4_net, _ = pool.allocate()
        pool.release(v4_net)
        content = open(network._LOCK_FILE).read()
        assert "192.168.200.0/30" not in content

    def test_allocate_after_release_reuses_subnet(self):
        pool = network.SubnetPool("192.168.200.0/29")
        v4_net, _ = pool.allocate()
        pool.release(v4_net)
        v4_net_2, _ = pool.allocate()
        assert v4_net == v4_net_2

    def test_pool_exhaustion_raises(self):
        pool = network.SubnetPool("192.168.200.0/29")
        pool.allocate()
        pool.allocate()
        with pytest.raises(RuntimeError, match="No available subnets"):
            pool.allocate()

    def test_allocate_reaps_dead_entries(self):
        pool = network.SubnetPool("192.168.200.0/29")
        proc = subprocess.Popen(
            ["python3", "-c", "exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        _write_lock_file_from_entries(pool, [
            ("192.168.200.0/30", proc.pid, "m-dead"),
        ])

        v4_net, _ = pool.allocate()
        assert v4_net == ipaddress.ip_network("192.168.200.0/30")
        content = open(network._LOCK_FILE).read()
        assert "m-dead" not in content


class TestReapDead:
    """Tests for SubnetPool.reap_dead."""

    def test_reap_dead_removes_dead_entries(self):
        pool = network.SubnetPool("192.168.200.0/29")
        proc = subprocess.Popen(
            ["python3", "-c", "exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        _write_lock_file_from_entries(pool, [
            ("192.168.200.0/30", proc.pid, "m-dead"),
        ])

        reaped = pool.reap_dead()
        assert reaped == ["m-dead"]
        content = open(network._LOCK_FILE).read()
        assert "192.168.200.0/30" not in content

    def test_reap_dead_keeps_alive_entries(self):
        pool = network.SubnetPool("192.168.200.0/29")
        _write_lock_file_from_entries(pool, [
            ("192.168.200.0/30", os.getpid(), "m-alive"),
        ])

        reaped = pool.reap_dead()
        assert reaped == []
        content = open(network._LOCK_FILE).read()
        assert "192.168.200.0/30" in content
        assert "m-alive" in content
