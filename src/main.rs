use std::f32;

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
            for j in 0..(frame_size - 1 - k) {
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
fn main() {
    let sample_rate = 44100.0;
    let f0_hz = 220.0;
    let duration_sec = 0.05;
    let frame_size = (sample_rate * duration_sec) as usize;

    // simple sin wave
    let mut test_frame = vec![0.0; frame_size];
    for i in 0..frame_size {
        test_frame[i] = (2.0 * f32::consts::PI * f0_hz * (i as f32 / sample_rate as f32)).sin();
    }

    let min_f0_search = 70.0;
    let max_f0_search = 500.0;

    if let Some(detected_f0) = get_pitch_acf(&test_frame, sample_rate, min_f0_search, max_f0_search)
    {
        println!("Detected F0: {:.2} Hz", detected_f0)
    } else {
        println!("Pitch not detected");
    }
}
