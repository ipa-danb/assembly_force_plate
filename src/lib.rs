mod rosrust_msg {
    rosrust::rosmsg_include!(geometry_msgs / WrenchStamped, std_msgs / String);
}

mod rust_fn {
    use crate::rosrust_msg;

    use ndarray::{arr1, Array1};
    use numpy::ndarray::{ArrayViewD, ArrayViewMutD};

    pub fn call_function(
        v: rosrust_msg::geometry_msgs::WrenchStamped,
        // df: data_field_mod::DataField,
    ) {
        println!("In mod recv: {}, {}", v.wrench.force.x, v.wrench.force.y); //, df.x);
    }
}

use ndarray;

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pyclass, pymethods, pymodule, types::PyModule, PyResult, Python};

#[pyclass]
pub struct DataField {
    #[pyo3(get, set)]
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl DataField {
    pub fn call_function(
        &mut self,
        v: rosrust_msg::geometry_msgs::WrenchStamped,
        // df: data_field_mod::DataField,
    ) {
        println!("In mod recv: {}, {}", v.wrench.force.x, v.wrench.force.y); //, df.x);
                                                                             // Ok()
    }
    pub fn test(&mut self) {
        println!("Test:"); //, df.x);
                           // Ok()
    }
}

#[pymodule]
fn rust_subscriber(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "start_subscriber")]
    fn start_subscriber(x: &PyArrayDyn<f64>) {
        rosrust::init("talker");
        let mut array = unsafe { x.as_array_mut() };
        let mut df = DataField { x: 0, y: 0, z: 0 };
        let cf = |v: rosrust_msg::geometry_msgs::WrenchStamped| df.call_function(v);
        {
            df.test();
            let subscriber = rosrust::subscribe(
                "/wrench_test",
                100,
                cf, // df.call_function,
            )
            .unwrap();
        }
        rosrust::spin();
    }

    Ok(())
}
