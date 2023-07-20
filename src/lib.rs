mod rosrust_msg {
    rosrust::rosmsg_include!(geometry_msgs / WrenchStamped, std_msgs / String);
}

use numpy::ndarray::{s, Array2};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::{pyclass, pymodule, types::PyFunction, types::PyModule, PyResult, Python};
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
    pub count: i32,
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
        dur: f64,
        topic_names: Vec<String>,
        pyfun: Py<PyFunction>,
    ) -> Vec<Py<PyArray2<f64>>> {
        let mut df_list = Vec::new();
        let cb_mutex = Arc::new(Mutex::new(pyfun));

        let callback = cb_mutex.clone();

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
        let dur_nsecs = (dur * 1_000_000_000 as f64) as i64;
        let d = rosrust::Duration::from_nanos(dur_nsecs);

        Python::with_gil(|py| loop {
            let mut ret_array = Vec::new();
            for d in df_list.iter() {
                let vecr = d.clone();
                let mut vecrr = vecr.lock().unwrap();
                let vecrrr = vecrr.vec.clone();
                let subarray = vecrrr.slice(s![..vecrr.count, ..]);
                ret_array.push(PyArray2::from_array(py, &subarray).to_owned());
                vecrr.count = 0;
            }
            callback.lock().unwrap().call1(py, (ret_array,));
            py.allow_threads(|| rosrust::sleep(d));
        });

        let mut ret_array = Vec::new();
        for d in df_list.iter() {
            let vecr = d.clone();
            let vecrr = vecr.lock().unwrap();
            let vecrrr = vecrr.vec.clone();
            let subarray = vecrrr.slice(s![..vecrr.count, ..]);
            ret_array.push(PyArray2::from_array(py, &subarray).to_owned());
        }
        let r1 = ret_array.clone();
        Python::with_gil(|py| {
            callback.lock().unwrap().call1(py, (ret_array,));
        });

        r1
    }

    Ok(())
}
