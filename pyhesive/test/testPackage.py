try:
    import pyhesive

    version = pyhesive.__version__
except:
    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    print("============================== ERROR importing package ==============================")
    print("Listing installed packages:")
    print(installed_packages)
    raise
