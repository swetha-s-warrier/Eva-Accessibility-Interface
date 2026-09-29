import os
import sys
import subprocess
import webbrowser
import platform

def launch_browser():
    """Opens the default browser to an accessible landing page."""
    webbrowser.open("https://www.google.com", new=2)
    return {"status": "success", "action": "BROWSER", "message": "Browser launched"}

def launch_media():
    """Opens system default media player or media web client."""
    current_os = platform.system()
    if current_os == "Windows":
        os.system("start wmplayer")
    elif current_os == "Darwin":  # macOS
        subprocess.Popen(["open", "-a", "Music"])
    else:  # Linux
        subprocess.Popen(["xdg-open", "https://music.youtube.com"])
    return {"status": "success", "action": "MEDIA", "message": "Media player triggered"}

def launch_files():
    """Opens the default local file explorer."""
    current_os = platform.system()
    home_dir = os.path.expanduser("~")
    if current_os == "Windows":
        subprocess.Popen(["explorer", home_dir])
    elif current_os == "Darwin":
        subprocess.Popen(["open", home_dir])
    else:
        subprocess.Popen(["xdg-open", home_dir])
    return {"status": "success", "action": "FILES", "message": "File explorer opened"}

def launch_notes():
    """Opens default text editor or local scratchpad."""
    current_os = platform.system()
    if current_os == "Windows":
        subprocess.Popen(["notepad.exe"])
    elif current_os == "Darwin":
        subprocess.Popen(["open", "-a", "TextEdit"])
    else:
        subprocess.Popen(["gedit"])
    return {"status": "success", "action": "NOTES", "message": "Notes opened"}

def launch_camera():
    """Invokes system camera preview utility or diagnostics viewer."""
    current_os = platform.system()
    if current_os == "Windows":
        subprocess.Popen(["start", "microsoft.windows.camera:"], shell=True)
    elif current_os == "Darwin":
        subprocess.Popen(["open", "-a", "Photo Booth"])
    else:
        subprocess.Popen(["cheese"])
    return {"status": "success", "action": "CAMERA", "message": "Camera launched"}