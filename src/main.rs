use std::net::{SocketAddrV4, UdpSocket};

use clap::{App, Arg};
use i2cdev::linux::LinuxI2CError;
use linux_embedded_hal::{Delay, I2cdev};
use mpu6050::{Mpu6050, Mpu6050Error, Steps};
use serde::{Deserialize, Serialize};
use serde_json;

type Result<T> = std::result::Result<T, Mpu6050Error<LinuxI2CError>>;

#[derive(Debug, Deserialize, Serialize)]
struct ThreeVector {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct Measurement {
    roll: f32,
    pitch: f32,
    temp: f32,
    gyro: ThreeVector,
    acc: ThreeVector,
}

impl Measurement {
    fn new(mpu: &mut Mpu6050<I2cdev, Delay>, steps: Option<u8>) -> Result<Measurement> {
        let rp = match steps {
            Some(steps) => mpu.get_acc_angles_avg(Steps(steps))?,
            None => mpu.get_acc_angles()?,
        };

        let temp = match steps {
            Some(steps) => mpu.get_temp_avg(Steps(steps))?,
            None => mpu.get_temp()?,
        };

        let gyro = match steps {
            Some(steps) => mpu.get_gyro_avg(Steps(steps))?,
            None => mpu.get_gyro()?,
        };

        let acc = match steps {
            Some(steps) => mpu.get_acc_avg(Steps(steps))?,
            None => mpu.get_acc()?,
        };

        let roll = rp.x;
        let pitch = rp.y;
        let gyro = ThreeVector {
            x: gyro.x,
            y: gyro.y,
            z: gyro.z,
        };
        let acc = ThreeVector {
            x: acc.x,
            y: acc.y,
            z: acc.z,
        };

        Ok(Measurement {
            roll,
            pitch,
            temp,
            gyro,
            acc,
        })
    }
}

fn main() -> Result<()> {
    let matches = App::new("MPU 6050 Test")
        .arg(
            Arg::with_name("udp")
                .short("u")
                .long("udp")
                .takes_value(true)
                .required(true)
                .help("UDP endpoint to log data to."),
        )
        .get_matches();

    let udp_addr: SocketAddrV4 = matches
        .value_of("udp")
        .unwrap()
        .parse()
        .expect("--udp value must be IP:PORT");

    let udp_sender = UdpSocket::bind("0.0.0.0:0").unwrap();

    let i2c = I2cdev::new("/dev/i2c-1").map_err(Mpu6050Error::I2c)?;
    let delay = Delay;
    let mut mpu = Mpu6050::new(i2c, delay);

    mpu.init()?;
    mpu.soft_calib(Steps(100))?;
    mpu.calc_variance(Steps(50))?;

    println!("Calibrated with bias: {:?}", mpu.get_bias().unwrap());
    println!("Calculated variance: {:?}", mpu.get_variance().unwrap());
    println!("");
    println!("Logging sensor measurements to UDP address: {}", udp_addr);

    loop {
        let measurement = Measurement::new(&mut mpu, None)?;
        let json_meas = serde_json::to_vec(&measurement).unwrap();
        udp_sender
            .send_to(&json_meas, udp_addr)
            .expect(&format!("Failed to send to UDP address: {}", udp_addr));
    }
}
