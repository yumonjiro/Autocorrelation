use anyhow::{Context, Result};
use std::path::Path;
use std::{env, f32, vec};

fn get_pitch_acf(frame_data: &[f32], sample_rate: f32, min_f0: f32, max_f0: f32) -> Option<f32> {
    let frame_size = frame_data.len();

    let min_lag = (sample_rate / max_f0).round() as usize;
    let mut max_lag = (sample_rate / min_f0).round() as usize;

    if max_lag >= frame_size {
        max_lag = frame_size - 1;
    }
    let mut autocorrelations = vec![0.0; max_lag + 1];
    // R[0]の計算
    for i in 0..frame_size {
        autocorrelations[0] += frame_data[i] * frame_data[i];
    }

    // R[k]の計算
    for k in 1..=max_lag {
        let mut sum = 0.0;
        if k > min_lag {
            for j in 0..(frame_size - k) {
                sum += frame_data[j] * frame_data[j + k];
            }
        }
        autocorrelations[k] = sum;
    }

    // ピーク探索
    let mut best_lag = 0;
    let mut max_val = -1.0f32;

    let r0 = autocorrelations[0];
    if r0 > 1e-6 {
        for k in min_lag..=max_lag {
            let normalized_val = autocorrelations[k] / r0;
            if normalized_val > max_val {
                max_val = autocorrelations[k] / r0;
                best_lag = k;
            }
        }
    } else {
        println!("?");
        return None;
    }

    let voicing_threshold = 0.3;

    if best_lag > 0 && max_val > voicing_threshold {
        let pitch_f0 = sample_rate / best_lag as f32;
        Some(pitch_f0)
    } else {
        None
    }
}

fn load_wav_to_f32(path: &Path) -> Result<(Vec<f32>, hound::WavSpec)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {:?}", path))?;
    let spec = reader.spec();

    if spec.channels != 1 {
        anyhow::bail!(
            "Only mono WAV files are supported. This file has {} channels.",
            spec.channels
        );
    }

    let samples: Result<Vec<_>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => reader
                    .samples::<i16>()
                    .map(|s| s.map(|x| x as f32 / 32768.0))
                    .collect(),
                24 => reader
                    .samples::<i32>() // houndはi24をi32として読む
                    .map(|s| s.map(|x| (x >> 8) as f32 / 8388608.0)) // i24 -> f32 normalized
                    .collect(),
                32 => reader // i32 の場合
                    .samples::<i32>()
                    .map(|s| s.map(|x| x as f32 / 2147483648.0))
                    .collect(),
                _ => anyhow::bail!(
                    "Unsupported bit depth: {} bits per sample for Int format",
                    spec.bits_per_sample
                ),
            }
        }
    };

    samples
        .map(|s| (s, spec))
        .with_context(|| format!("Fasiled to read samples from WAV file"))
}

fn apply_hanning_window(frame: &[f32]) -> Vec<f32> {
    let frame_len = frame.len();
    if frame_len == 0 {
        return Vec::new();
    }
    let mut windowed_frame = Vec::with_capacity(frame_len);
    for i in 0..frame_len {
        let multiplier =
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / (frame_len - 1) as f32).cos());
        windowed_frame.push(frame[i] * multiplier);
    }
    windowed_frame
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_wav_file", args[0]);
        std::process::exit(1);
    }
    let wav_path = Path::new(&args[1]);

    // Parameters
    let frame_duration_ms = 30.0;
    let hop_duration_ms = 10.0;
    let min_f0_hz = 80.0;
    let max_f0_hz = 1000.0;

    // Read wav file
    let (audio_data, spec) = load_wav_to_f32(wav_path)?;
    let sample_rate = spec.sample_rate as f32;

    println!(
        "Loaded WAV: {:?}, Sample Rate: {} Hz, Duration: {:.2}s",
        wav_path.file_name().unwrap_or_default(),
        sample_rate,
        audio_data.len() as f32 / sample_rate
    );
    let frame_size_samples = (sample_rate * frame_duration_ms / 1000.0).round() as usize;
    let hop_size_samples = (sample_rate * hop_duration_ms / 1000.0).round() as usize;

    if frame_size_samples == 0 || hop_size_samples == 0 {
        anyhow::bail!("Frame size or hop size is too small for the given sample rate.");
    }
    println!(
        "Frame size: {} samples ({:.1} ms)",
        frame_size_samples, frame_duration_ms
    );
    println!(
        "Hop size: {} samples ({:.1} ms)",
        hop_size_samples, hop_duration_ms
    );
    println!("--------------------------------------------------");
    println!("Time (s)\tDetected F0 (Hz)");
    println!("--------------------------------------------------");

    // --- フレーム処理とピッチ検出 ---
    let mut current_pos = 0;
    while current_pos + frame_size_samples <= audio_data.len() {
        let frame = &audio_data[current_pos..current_pos + frame_size_samples];

        let windowed_frame = apply_hanning_window(frame);
        let detected_f0 = get_pitch_acf(&windowed_frame, sample_rate, min_f0_hz, max_f0_hz);

        let time_sec = current_pos as f32 / sample_rate;

        match detected_f0 {
            Some(f0) => println!("{:.3}\t\t{:.2}", time_sec, f0),
            None => println!("{:.3}\t\t-- (Unvoiced/Silent)", time_sec),
        }

        current_pos += hop_size_samples;
    }
    println!("--------------------------------------------------");
    println!("Processing finished.");

    Ok(())
}
