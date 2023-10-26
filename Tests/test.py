from datetime import datetime

now = datetime.now()

current_time = now.strftime("%Y_%m_%d_%H_%M")
print("Current Time =", current_time)

# now.strftime("%Y-%m-%d %H:%M:%S")