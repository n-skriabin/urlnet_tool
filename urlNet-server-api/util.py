import sys

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def exit_on_busy_port():
    if is_port_in_use(5000):
        print("PORT IS BUSY (5000)! CHECK TASK MANAGER")
        sys.exit()