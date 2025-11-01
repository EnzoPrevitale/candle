use candle_core::{Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x: Tensor = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second) 
    }
}

fn main() -> Result<()> {
    let device: Device = Device::Cpu; // Usando o processador

    let first: Tensor = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second: Tensor = Tensor::randn(0f32, 1.0, (100, 10), &device)?;

    let model: Model = Model { first, second };

    let dummy_image: Tensor = Tensor::randn(0f32, 1.0, (1,784), &device)?;

    let digit: Tensor = model.forward(&dummy_image)?;
    println!("Digit: {digit:?}");


    Ok(())
}