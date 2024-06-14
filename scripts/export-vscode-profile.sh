#!/bin/bash

# Export the list of installed extensions
code --list-extensions > .vscode/extensions.list

# Copy the global user settings to the project-specific file
cp $HOME/.config/Code/User/settings.json .vscode/user_settings.json
