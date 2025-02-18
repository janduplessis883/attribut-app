import streamlit as st
import sys
import time

class StreamlitStdout:
    def __init__(self, container):
        self.container = container
        self.lines = []  # Store each complete line

    def write(self, text):
        # When text contains a newline, split it into lines
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line.strip())
                # Update the container with all lines joined (each printed only once)
                self.container.code("\n".join(self.lines))

    def flush(self):
        pass  # no-op for compatibility

# Create a container for terminal output using st.empty()
terminal_container = st.empty()

# Redirect stdout to our custom handler
old_stdout = sys.stdout
sys.stdout = StreamlitStdout(terminal_container)

# Simulate some process with print statements
print("Hello, this is terminal output captured by Streamlit!")
time.sleep(1)
print("⏰ Scheduling 'Transformation Fund Needs to be submitted' at 2025-02-18 09:45:00 until 2025-02-18 10:15:00")
time.sleep(1)
print("✅ Task 'CKD Study Searches' (ID: 19dfdfd6-8a97-8078-8e7c-c3d6b1f43758) updated successfully.")
time.sleep(1)
print("🤖 NN predicts extra time of 0.00 min for task 'CKD Study Searches'")
time.sleep(1)
print("Hello, this is terminal output captured by Streamlit!")
