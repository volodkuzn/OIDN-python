# OIDN-python command line tool

import oidn

if __name__ == "__main__":
    with oidn.Device("cpu") as device, oidn.Filter(device, "RT") as filter:
        pass
