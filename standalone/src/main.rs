mod rosrust_msg {
    rosrust::rosmsg_include!(
        geometry_msgs / WrenchStamped,
        std_msgs / String,
        geometry_msgs / Twist,
        geometry_msgs / Vector3
    );
}

use ndarray::Array2;
use std::sync::Arc;
use std::sync::Mutex;

pub struct DataField {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub vec: Array2<f64>,
    pub tres: i32,
    pub count: i32,
    pub alpha: f64,
    pub used: bool,
}

pub struct SensorKit {
    pub vec: Vec<DataField>,
    pub data_vec: Array2<f64>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub tres: i32,
    pub count: i32,
    pub x1: f64,
    pub x2: f64,
    pub h: f64,
    pub publisher: rosrust::Publisher<rosrust_msg::geometry_msgs::Twist>,
}

impl SensorKit {
    pub fn calc_forces(&mut self) {
        if self.vec.iter().any(|x| x.used == false) {
            return;
        }
        if self.count >= self.tres {
            self.count = 0;
        }
        let denom = self.vec[0].z + self.vec[1].z + self.vec[2].z + self.vec[3].z;
        self.y = self.x1
            * 0.5
            * ((self.vec[0].z + self.vec[1].z - self.vec[2].z - self.vec[3].z)
                + self.h * (self.vec[0].y + self.vec[1].y + self.vec[2].y + self.vec[3].y))
            / denom;
        self.x = self.x2
            * 0.5
            * ((self.vec[0].z - self.vec[1].z - self.vec[2].z + self.vec[3].z)
                + self.h * (self.vec[0].x + self.vec[1].x + self.vec[2].x + self.vec[3].x))
            / denom;

        self.data_vec[[self.count as usize, 1]] = self.x; //v.wrench.force.x;
        self.data_vec[[self.count as usize, 2]] = self.y;

        for v in self.vec.iter_mut() {
            v.used = false;
        }
        self.count += 1;

        self.pub_forces();

        // println!("x:{:.4}, y: {:.4}", self.x, self.y);
    }
    pub fn pub_forces(&mut self) {
        let mut msg = rosrust_msg::geometry_msgs::Twist::default();
        msg.linear.x = self.x + self.x2 / 2.0;
        msg.linear.y = self.y + self.x1 / 2.0;
        self.publisher.send(msg).unwrap();
    }
}

fn alpha_filter(new_value: f64, old_value: f64, alpha: f64) -> f64 {
    return new_value * alpha + old_value * (1.0 - alpha);
}

impl DataField {
    pub fn call_function(&mut self, v: rosrust_msg::geometry_msgs::WrenchStamped) {
        if self.count >= self.tres {
            self.count = 0;
            println!("Test: {:?}", self.vec);
        }

        self.x = alpha_filter(v.wrench.force.x, self.x, self.alpha);
        self.y = alpha_filter(v.wrench.force.y, self.y, self.alpha);
        self.z = alpha_filter(v.wrench.force.z, self.z, self.alpha);
        self.tx = alpha_filter(v.wrench.torque.x, self.tx, self.alpha);
        self.ty = alpha_filter(v.wrench.torque.y, self.ty, self.alpha);
        self.tz = alpha_filter(v.wrench.torque.z, self.tz, self.alpha);

        self.vec[[self.count as usize, 0]] = v.header.stamp.seconds();

        self.vec[[self.count as usize, 1]] = self.x; //v.wrench.force.x;
        self.vec[[self.count as usize, 2]] = self.y;
        self.vec[[self.count as usize, 3]] = self.z;

        self.vec[[self.count as usize, 4]] = self.tx;
        self.vec[[self.count as usize, 5]] = self.ty;
        self.vec[[self.count as usize, 6]] = self.tz;
        self.used = true;
        self.count += 1;
    }
}

fn main() {
    rosrust::init("talkert");
    let df_list = Vec::new();
    let mut topic_names: Vec<String> = Vec::new();

    let xy_pub: rosrust::Publisher<rosrust_msg::geometry_msgs::Twist> =
        rosrust::publish("/xy", 100).unwrap();

    topic_names.push("/wrench_1".to_string());
    topic_names.push("/wrench_2".to_string());
    topic_names.push("/wrench_3".to_string());
    topic_names.push("/wrench_4".to_string());
    println!("STart");

    // Fill in new Datafields per topic
    let sensorkit = Arc::new(Mutex::new(SensorKit {
        vec: df_list,
        x: 0.,
        y: 0.,
        z: 0.,
        tres: 1000,
        count: 0,
        x1: 0.38,
        x2: 0.38,
        h: 0.02,
        data_vec: Array2::<f64>::zeros((1000 as usize, 7)),
        publisher: xy_pub,
    }));

    for _d in topic_names.iter() {
        let mut sensorkit_unlocked = sensorkit.lock().unwrap();
        sensorkit_unlocked.vec.push(DataField {
            x: 0.,
            y: 0.,
            z: 0.,
            tx: 0.,
            ty: 0.,
            tz: 0.,
            tres: 1000,
            alpha: 0.3,
            count: 0,
            used: false,
            vec: Array2::<f64>::zeros((1000 as usize, 7)),
        })
    }

    // Start the subscribers
    println!("Subscribe");

    let mut subscriber = Vec::new();
    for (i, d) in topic_names.iter().enumerate() {
        let sensorkit_cloned = sensorkit.clone();
        subscriber.push(
            rosrust::subscribe(
                d,
                100,
                move |v: rosrust_msg::geometry_msgs::WrenchStamped| {
                    let mut sensorkit_locked = sensorkit_cloned.lock().unwrap();
                    sensorkit_locked.vec[i].call_function(v);
                    sensorkit_locked.calc_forces();
                },
            )
            .unwrap(),
        );
    }
    println!("spin");

    rosrust::spin();
}
