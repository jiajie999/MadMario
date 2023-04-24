# When running on server without display, we need a virtual device.
xvfb-run -s "-screen 0 1400x900x24" python main.py
