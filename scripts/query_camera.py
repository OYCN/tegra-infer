import subprocess
import os

class CmdException(Exception): pass

def run(cmd, check=False):
    ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ret.wait()
    txt = ret.stdout.read().decode('utf-8')
    success = ret.returncode != 0
    if check and success:
        raise CmdException(txt)
    return (success, txt)

def has_v4l2_utils():
    _, ret = run(f'whereis v4l2-ctl', True)
    path = ret.split(':')[-1].strip()
    return os.path.exists(path)

def get_all_device():
    _, ret = run(f'v4l2-ctl --list-devices', True)
    devices = {}
    ptr = None
    for line in ret.split('\n'):
        line = line.strip()
        # or can be check the nb of '\t'
        if line.startswith('/dev/'):
            assert ptr is not None
            ptr.append(line)
        elif len(line) == 0:
            ptr = None
        else:
            devices[line] = []
            ptr = devices[line]
    return devices

def query_info(dev):
    _, ret = run(f'v4l2-ctl --device={dev} --list-formats-ext')
    info_map = {}
    size_map = None
    fps_list = None
    if '[0]' not in ret:
        return {}
    ret = ret.split('\n')[3:]
    for line in ret:
        if line.startswith('\t\t\t'):
            assert fps_list is not None
            fps = float(line.strip().split('(')[-1].split(' ')[0])
            fps_list.append(fps)
        elif line.startswith('\t\t'):
            assert size_map is not None
            a, b = line.strip().split(' ')[-1].split('x')
            size = (int(a), int(b))
            size_map[size] = []
            fps_list = size_map[size]
        elif line.startswith('\t'):
            format = line.strip().split(']:')[-1].strip()
            info_map[format] = {}
            size_map = info_map[format]
    return info_map


if __name__ == '__main__':
    if not has_v4l2_utils():
        print(f'v4l2-ctl not found, please install by `sudo apt install v4l-utils`')
        exit(1)
    devices = get_all_device()
    all_map = {}
    for dn, dp_s in devices.items():
        all_map[dn] = {}
        this_map = all_map[dn]
        for dp in dp_s:
            if dp.startswith('/dev/video'):
                info = query_info(dp)
                for format, v in info.items():
                    format = format.split("'")[1]
                    for size, vv in v.items():
                        for fps in vv:
                            key = (format, size, fps)
                            if key not in this_map:
                                this_map[key] = set()
                            this_map[key].add(dp)
    for dev, v in all_map.items():
        print(dev)
        for info, path in v.items():
            print(f'\t{info[0]} {info[1][0]}x{info[1][1]} {info[2]}: {path}')

