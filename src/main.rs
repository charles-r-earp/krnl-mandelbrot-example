use clap::Parser;
use image::GrayImage;
use krnl::macros::module;
use std::{io::Write, time::Instant};

fn naive(h: u32, w: u32, max_iterations: u32) -> Vec<u8> {
    (0..h)
        .flat_map(move |r| (0..w).map(move |c| kernels::mandelbro_impl(r, c, h, w, max_iterations)))
        .collect()
}

fn parallel(h: u32, w: u32, max_iterations: u32) -> Vec<u8> {
    use rayon::prelude::*;

    (0..h)
        .into_par_iter()
        .flat_map_iter(move |r| {
            (0..w).map(move |c| kernels::mandelbro_impl(r, c, h, w, max_iterations))
        })
        .collect()
}

fn gpu(index: usize, h: u32, w: u32, max_iterations: u32) -> Vec<u8> {
    use krnl::{buffer::Buffer, device::Device};

    let device = Device::builder().index(index).build().unwrap();
    let kernel = kernels::mandelbrot::builder()
        .unwrap()
        .specialize(h, w, max_iterations)
        .build(device.clone())
        .unwrap();
    let mut y = Buffer::zeros(device.clone(), (h * w) as usize).unwrap();
    let device_info = device.info().unwrap();
    // On some devices max_groups is too small, so split operation into multiple dispatches.
    let global_threads =
        (device_info.max_groups() as usize * device_info.default_threads() as usize).min(y.len());
    for offset in (0..y.len()).step_by(global_threads) {
        let end = (offset + global_threads).min(y.len());
        if let Some(y) = y.slice_mut(offset..end) {
            kernel.dispatch(y, offset as u32).unwrap();
        } else {
            break;
        }
    }
    y.to_vec().unwrap()
}

#[module]
mod kernels {

    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::macros::kernel;

    pub(crate) fn iterations_to_grayscale(i: u32, max_iterations: u32) -> u8 {
        #[cfg(target_arch = "spirv")]
        use krnl_core::num_traits::Float;

        if i == max_iterations {
            return 0;
        }
        (i as f32 * 255f32 / max_iterations as f32).round() as u8
    }

    pub(crate) fn mandelbro_impl(r: u32, c: u32, h: u32, w: u32, max_iterations: u32) -> u8 {
        let x0 = ((c as f32) / (w as f32)) * 3.5 - 2.5;
        let y0 = ((r as f32) / (h as f32)) * 2.0 - 1.0;
        let mut x = 0f32;
        let mut y = 0f32;
        let mut iteration = 0;
        while x * x + y * y <= 4.0 && iteration < max_iterations {
            let xtemp = x * x - y * y + x0;
            y = 2.0 * x * y + y0;
            x = xtemp;
            iteration += 1;
        }
        iterations_to_grayscale(iteration, max_iterations)
    }

    #[kernel]
    pub(crate) fn mandelbrot<const H: u32, const W: u32, const I: u32>(
        #[item] y: &mut u8,
        offset: u32,
    ) {
        let idx = offset + kernel.item_id() as u32;
        let r = idx / W;
        let c = idx % W;
        *y = mandelbro_impl(r, c, H, W, I);
    }
}

fn runalgo(
    name: &str,
    h: u32,
    w: u32,
    max_iterations: u32,
    save_image: bool,
    algo: impl Fn(u32, u32, u32) -> Vec<u8>,
) {
    print!("Executing {}... ", name);
    std::io::stdout().flush().unwrap();
    let now = Instant::now();
    let img = GrayImage::from_vec(w, h, algo(h, w, max_iterations)).unwrap();
    let elapsed = now.elapsed();
    if save_image {
        let fname = format!("mandelbrot_{name}.png");
        img.save_with_format(&fname, image::ImageFormat::Png)
            .unwrap();
        println!("Saved image to {fname:?}.");
    }
    println!("{elapsed:.1?}");
}

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    naive: bool,
    #[arg(long)]
    parallel: bool,
    #[arg(long)]
    gpu: Option<usize>,
    /// Default if no algorithm is specified.
    #[arg(long)]
    all: bool,
    #[arg(long, default_value_t = 8_000)]
    height: u32,
    #[arg(long, default_value_t = 14_000)]
    width: u32,
    #[arg(long, default_value_t = 1000)]
    max_iterations: u32,
    #[arg(long)]
    save: bool,
}

fn main() {
    let cli = Cli::parse();
    let all = !(cli.naive || cli.parallel || cli.gpu.is_some());
    let height = cli.height;
    let width = cli.width;
    let max_iterations = cli.max_iterations;
    let save_image = cli.save;
    if cli.naive || all {
        runalgo("naive", height, width, max_iterations, save_image, naive);
    }
    if cli.parallel || all {
        runalgo(
            "parallel",
            height,
            width,
            max_iterations,
            save_image,
            parallel,
        );
    }
    if cli.gpu.is_some() || all {
        let index = cli.gpu.unwrap_or(0);
        runalgo(
            "gpu",
            height,
            width,
            max_iterations,
            save_image,
            |h, w, max_iterations| gpu(index, h, w, max_iterations),
        );
    }
}
