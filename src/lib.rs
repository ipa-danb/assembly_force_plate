mod rosrust_msg {
    rosrust::rosmsg_include!(geometry_msgs / WrenchStamped, std_msgs / String);
}

use numpy::ndarray::{s, Array2};
use numpy::PyArray2;
use pyo3::{pyclass, pymodule, types::PyModule, PyResult, Python};
use std::sync::Arc;
use std::sync::Mutex;

#[pyclass]
pub struct DataField {
    #[pyo3(get, set)]
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vec: Array2<f64>,
    pub tres: i32,
    count: i32,
}

impl DataField {
    pub fn call_function(&mut self, v: rosrust_msg::geometry_msgs::WrenchStamped) {
        if self.count >= self.tres {
            self.count = 0;
        }
        self.x = v.wrench.force.x;
        self.y = v.wrench.force.y;
        self.z = v.wrench.force.z;
        self.vec[[self.count as usize, 0]] = v.header.stamp.seconds();

        self.vec[[self.count as usize, 1]] = v.wrench.force.x;
        self.vec[[self.count as usize, 2]] = v.wrench.force.y;
        self.vec[[self.count as usize, 3]] = v.wrench.force.z;

        self.vec[[self.count as usize, 4]] = v.wrench.torque.x;
        self.vec[[self.count as usize, 5]] = v.wrench.torque.y;
        self.vec[[self.count as usize, 6]] = v.wrench.torque.z;

        self.count += 1;
    }
}
use pyo3::prelude::*;

#[pymodule]
fn rust_subscriber(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataField>()?;
    #[pyfn(m)]
    #[pyo3(name = "start_node")]
    fn start_node() {
        rosrust::init("talkert");
    }
    #[pyfn(m)]
    #[pyo3(name = "start_subscriber")]
    fn start_subscriber<'py>(
        py: Python<'py>,
        buf_size: i32,
        dur: i32,
        topic_names: Vec<String>,
    ) -> Vec<Py<PyArray2<f64>>> {
        let mut df_list = Vec::new();
        for _d in topic_names.iter() {
            df_list.push(Arc::new(Mutex::new(DataField {
                x: 0.,
                y: 0.,
                z: 0.,
                tres: buf_size,
                count: 0,
                vec: Array2::<f64>::zeros((buf_size as usize, 7)),
            })))
        }

        // let df = Mutex::new(df);
        let mut subscriber = Vec::new();
        for (i, d) in df_list.iter().enumerate() {
            let dfr = d.clone();
            subscriber.push(
                rosrust::subscribe(
                    &topic_names[i], //"/wrench_test",
                    100,
                    move |v: rosrust_msg::geometry_msgs::WrenchStamped| {
                        let mut dfrr = dfr.lock().unwrap();
                        dfrr.call_function(v);
                    }, // df.call_function,
                )
                .unwrap(),
            );
        }

        let d = rosrust::Duration { sec: dur, nsec: 0 };
        rosrust::sleep(d);
        let mut ret_array = Vec::new();
        for d in df_list.iter() {
            let vecr = d.clone();
            let vecrr = vecr.lock().unwrap();
            let vecrrr = vecrr.vec.clone();
            let subarray = vecrrr.slice(s![..vecrr.count, ..]);
            ret_array.push(PyArray2::from_array(py, &subarray).to_owned());
        }

        ret_array
    }

    Ok(())
}
