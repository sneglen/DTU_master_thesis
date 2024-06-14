import time
import os

# Create a flag file to indicate the server is running
print("LLM server starting...")

flag_file = "server_running.flag"

try:
  open(flag_file, 'w').close()
except Exception as e:
  print(f"Error creating {flag_file}: {e}")

init_time = time.time()
elapsed_time = 0

try:
    while elapsed_time < 60:
        elapsed_time = time.time() - init_time
        print(f"[{elapsed_time:.2f}]: LLM server running...")
        time.sleep(2)

finally:
    # Clean up the flag file when the server stops
    print("LLM server finished.")
    os.remove(flag_file)
