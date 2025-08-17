# teach_tool_test

# Start Mockup Ft Sensor
Driver:
`roslaunch teach_tool test.launch sim:=True`
Mockup Test GUI:
`rosrun teach_tool mockup_test_gui.py`




# Start with real Sensors
`roslaunch teach_tool_test test.launch sim:=False`

## 

* `pdm run maturin --develop` makes development target available in venv
* `pdm run maturin build -r --manylinux off`
* install with `cd target/wheels && pip install rust_subscriber....py`


