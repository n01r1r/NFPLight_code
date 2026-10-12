"""Stop before checkpoint writes exhaust the workspace disk."""
import shutil

class DiskReserveError(OSError):
    pass

def require_disk_reserve(directory):
    free = shutil.disk_usage(directory).free
    if free < 5 * 1024**3:
        raise DiskReserveError(f"checkpoint reserve below 5 GiB: {free} bytes free")
    return free
