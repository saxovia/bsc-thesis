import sys, os, subprocess

def check_python_version():
    major, minor, _ = sys.version_info[:3]
    if major==3 and minor>9:
        print("Your Python version is too new. This script requires Python 3.9 due to PyQt6's libraries' unfixed bugs.")
        sys.exit(1)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_requirements():
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        with open(req_file) as f:
            packages = f.read().splitlines()
        for pkg in packages:
            if pkg and not pkg.startswith("#"):
                print("Installing " + pkg)
                install(pkg)
    else:
        print(req_file + " not found!")

if __name__=="__main__":
    check_python_version()
    install_requirements()
