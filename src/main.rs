use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device: Device = Device::Cpu;

    let a: Tensor = Tensor::new(&[1f32, 2., 3., 4.], &device)?.reshape((2, 2))?;
    let b: Tensor = Tensor::new(&[5f32, 6., 7., 8.], &device)?.reshape((2, 2))?;

    let c: Tensor = a.matmul(&b)?;
    println!("{c}\n\n");

    let c: Tensor = a.t()?;
    println!("{c}\n\n");

    let c: Tensor = a.add(&b)?;
    println!("{c}\n\n");

    let c: Tensor = a.sum_all()?;
    println!("{c}\n\n");

    let c: Tensor = a.sum_all()?;
    println!("{c}\n\n");    

    let c: Tensor = a.mean(0)?;
    println!("{c}\n\n");

    let c: Tensor = a.broadcast_add(&b)?;
    println!("{c}\n\n");

    let c: Tensor = a.broadcast_mul(&b)?;
    println!("{c}\n\n");

    Ok(())
}