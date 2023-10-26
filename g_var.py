def _init():  # 初始化
    global g_iteration
    g_iteration = 0

def set_value(value):
    global g_iteration
    g_iteration = value

def get_value():
    global g_iteration
    return g_iteration