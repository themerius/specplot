"""Example usage of specplot."""

from specplot import diagram, node, PathfindingConfig


def sandwich_example():
    """Create a sandwich layout example (3 rows: 1 zone, 3 zones, 1 zone)."""
    with diagram(
        filename="output/sandwich_example",
        layout=(("LR",), ("TB", "TB", "TB"), ("LR",))
    ):
        # pos=1: top LR zone
        user = node(icon="person", label="User", pos=1)

        # pos=2,3,4: middle TB columns
        api = node(icon="api", label="API Gateway", pos=2)
        auth = node(icon="lock", label="Auth Service", pos=2)

        svc = node(icon="dns", label="Core Service", pos=3)
        cache = node(icon="memory", label="Cache", pos=3)

        db = node(icon="database", label="Database", pos=4)
        backup = node(icon="backup", label="Backup", pos=4)

        # pos=5: bottom LR zone
        logs = node(icon="description", label="Logs", pos=5)
        metrics = node(icon="monitoring", label="Metrics", pos=5)

        # Edges
        user >> api >> svc >> db
        api >> auth
        svc >> cache
        db >> backup
        svc >> logs
        svc >> metrics

    print("Sandwich diagram saved to sandwich_example.svg")


def main():
    """Create an example architecture diagram."""
    with diagram(filename="output/example"):
        # Simple node - no 'with' needed when no children
        user = node(icon="person", label="User")

        # Group with nested nodes - use 'with' for children
        with node(
            icon="hive",
            label="Swarm Colony",
            description="Contains essential enviornment for agents to work.",
        ) as env:
            cli = node(icon="terminal", label="Command Line Interface")
            goal = node(icon="assignment", label="Overall Goal")
            db = node(icon="database", label="Data Store", description="hello db")
            dbf = node(icon="database", label="Failure Store", description="hello db")
            dbs = node(icon="database", label="Signals", description="hello db")

        # Outline mode (default) - use 'with' for children
        with node(
            icon="genetics",
            label="Swarm Agents",
            description="All agents implement this interface.",
            show_as="group",
            grid=(3, 2),
        ) as agents:
            node(icon="psychology", label="Head", description="This thing will parse user input and translate it to machine readable inputs.")
            node(icon="psychology", label="Email2PDF")
            node(icon="psychology", label="PDF2Markdown")
            node(icon="psychology", label="Classification", description="Detects if invoice")
            with node(icon="psychology", label="Annotate", description="Tagging of invoice infos"):
                node(icon="psychology", label="sub agent 1")
                node(icon="psychology", label="sub agent 2")
            writer = node(icon="psychology", label="Writer")

        node(icon="cloud_download", label="IMAP service")
        node(icon="extension", label="docling lib")
        node(icon="graph_5", label="LLM (4b)")
        node(icon="graph_5", label="LLM (1.7b)")
        fs = node(icon="folder_open", label="Filesystem")

        # Edges
        user >> cli
        agents >> env | "uses"
        writer >> fs | "writes"

    print("Diagram saved to example.svg")


def linux_kernel_example():
    """Create a Linux kernel architecture diagram using groups and outlines.

    Based on the conceptual architecture with 5 main subsystems:
    - Process Scheduler (SCHED) - central, all others depend on it
    - Memory Manager (MM) - virtual/physical memory management
    - Virtual File System (VFS) - abstracts file systems
    - Network Interface (NET) - networking protocols
    - Inter-Process Communication (IPC) - process messaging

    Layout: User space -> Kernel (with grouped subsystems) -> Hardware
    """
    config = PathfindingConfig(debug=True)
    with diagram(
        filename="output/linux_kernel",
        pathfinding=config,
        layout=(
            ("LR",),              # User space (group)
            ("LR",),              # System call interface
            ("LR", "LR", "LR"),   # Kernel subsystems (3 groups)
            ("LR",),              # Device drivers (group)
            ("LR",),              # Hardware (group)
        )
    ):
        # === Row 1: User Space as a group ===
        with node(
            icon="person",
            label="User Space",
            description="Ring 3 - unprivileged mode",
            show_as="group",
            grid=(1, 2),
            pos=1
        ):
            apps = node(icon="apps", label="Applications",
                       description="Shells, editors, browsers, daemons")
            libs = node(icon="library_books", label="C Library",
                       description="glibc: malloc, printf, pthread")

        # === Row 2: System Call Interface ===
        syscall = node(icon="swap_vert", label="System Call Interface",
                      description="read, write, fork, exec, mmap, socket...", pos=2)

        # === Row 3: Kernel Subsystems as groups ===
        # Group 1: File Systems (VFS + implementations)
        with node(
            icon="folder_special",
            label="Virtual File System",
            description="Unified file/device interface",
            show_as="group",
            grid=(2, 2),
            pos=3
        ) as vfs:
            ext4 = node(icon="storage", label="ext4",
                       description="Primary Linux filesystem")
            xfs = node(icon="storage", label="XFS",
                       description="High-performance journaling")
            nfs = node(icon="cloud_sync", label="NFS",
                       description="Network File System")
            proc = node(icon="info", label="procfs",
                       description="Process information")

        # Group 2: Process & Memory (central subsystems)
        with node(
            icon="developer_board",
            label="Process & Memory",
            description="Core kernel subsystems",
            show_as="group",
            grid=(2, 1),
            pos=4
        ):
            # Scheduler with outline children
            with node(
                icon="schedule",
                label="Process Scheduler",
                description="CPU time distribution, context switching",
                show_as="outline"
            ) as sched:
                cfs = node(label="CFS - Completely Fair Scheduler")
                rt = node(label="Real-time scheduling classes")
                idle = node(label="Idle task management")

            # Memory Manager with outline children
            with node(
                icon="memory",
                label="Memory Manager",
                description="Virtual memory, paging, allocation",
                show_as="outline"
            ) as mm:
                vm = node(label="Virtual memory (mmap, brk)")
                page = node(label="Page frame allocator")
                slab = node(label="Slab/SLUB allocator")

        # Group 3: Network & IPC
        with node(
            icon="lan",
            label="Network & IPC",
            description="Communication subsystems",
            show_as="group",
            grid=(2, 1),
            pos=5
        ):
            # Network with outline
            with node(
                icon="cloud",
                label="Network Stack",
                description="Protocol implementation",
                show_as="outline"
            ) as net:
                socket_api = node(label="Socket API (BSD sockets)")
                tcp_ip = node(label="TCP/IP protocol stack")
                netfilter = node(label="Netfilter (iptables/nftables)")

            # IPC with outline
            with node(
                icon="forum",
                label="IPC Subsystem",
                description="Inter-process communication",
                show_as="outline"
            ) as ipc:
                pipe = node(label="Pipes and FIFOs")
                shm = node(label="Shared memory (shmem)")
                signal = node(label="Signals and semaphores")

        # === Row 4: Device Drivers as outline ===
        with node(
            icon="settings_input_hdmi",
            label="Device Drivers",
            description="Hardware abstraction layer",
            show_as="outline",
            pos=6
        ) as drivers:
            char_drv = node(label="Character devices (tty, input)")
            block_drv = node(label="Block devices (SCSI, NVMe)")
            net_drv = node(label="Network drivers (e1000, iwlwifi)")

        # === Row 5: Hardware as group ===
        with node(
            icon="hardware",
            label="Hardware",
            description="Physical components - Ring 0 access",
            show_as="group",
            grid=(1, 4),
            pos=7
        ):
            cpu = node(icon="memory", label="CPU",
                      description="x86, ARM, RISC-V")
            ram = node(icon="sd_storage", label="RAM",
                      description="Physical memory")
            disk = node(icon="hard_drive", label="Storage",
                       description="SSD, HDD, NVMe")
            nic = node(icon="router", label="NIC",
                      description="Network interface")

        # === Edges ===
        # User space to syscall
        apps >> syscall
        libs >> syscall

        # Syscall to kernel subsystems
        syscall >> vfs
        syscall >> sched
        syscall >> net

        # Inter-subsystem dependencies
        mm >> sched | "suspend"
        vfs >> mm | "page cache"
        ipc >> mm | "shm"
        nfs >> net  # NFS uses network stack

        # Subsystems to drivers
        vfs >> drivers
        net >> drivers

        # Drivers to hardware
        drivers >> cpu
        drivers >> disk
        drivers >> nic
        mm >> ram

    print("Linux kernel diagram saved to linux_kernel.svg")


if __name__ == "__main__":
    main()
    sandwich_example()
    linux_kernel_example()
