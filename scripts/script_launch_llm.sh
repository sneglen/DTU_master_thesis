#!/bin/bash

# Create a new session named 'ss' if it doesn't exist
tmux has-session -t ss 2>/dev/null
if [ $? != 0 ]; then
  tmux new-session -d -s ss
fi

# Create a new window for launching the LLM, if it doesn't exist
if ! tmux list-windows -t ss | grep -q 'llm'; then
  tmux new-window -d -t ss -n llm
  tmux send-keys -t ss:llm 'cdMT' C-m
  tmux send-keys -t ss:llm 'eMT' C-m
  tmux send-keys -t ss:llm 'make launch_llm' C-m
fi

# Switch back to window 0
tmux attach-session -t ss:0