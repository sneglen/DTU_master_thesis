import time
import os

init_time = time.time()
elapsed_time = 0

while elapsed_time < 120:
    elapsed_time = time.time() - init_time
    try:
      # Check if the server is running by checking for the flag file
      server_running = os.path.exists("server_running.flag")
      server_status = "RUNNING" if server_running else "HALTED"
    except Exception as e:
      server_status = "ERROR (could not check <server_running.flag>)"

    print(f"[{elapsed_time:.2f}]: LLM client (making JSON files)... Server is {server_status}")
    time.sleep(2)
