/// Scalar field that can be visualized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarField {
    Velocity,
    Pressure,
    Density,
}

/// Available color map types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMapType {
    Viridis,
    Plasma,
    Coolwarm,
    Water,
}

/// Color map: maps a normalized scalar [0, 1] to an RGB color.
pub struct ColorMap;

impl ColorMap {
    /// Map a scalar value in [0, 1] to an [r, g, b] color using the given color map.
    pub fn map(t: f32, map_type: ColorMapType) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        match map_type {
            ColorMapType::Viridis => Self::viridis(t),
            ColorMapType::Plasma => Self::plasma(t),
            ColorMapType::Coolwarm => Self::coolwarm(t),
            ColorMapType::Water => Self::water(t),
        }
    }

    /// Viridis color map (simplified 5-stop approximation).
    fn viridis(t: f32) -> [f32; 3] {
        // Key stops: dark purple -> blue -> teal -> green -> yellow
        let colors: [(f32, [f32; 3]); 5] = [
            (0.0, [0.267, 0.004, 0.329]),
            (0.25, [0.282, 0.140, 0.458]),
            (0.5, [0.127, 0.566, 0.551]),
            (0.75, [0.369, 0.789, 0.383]),
            (1.0, [0.993, 0.906, 0.144]),
        ];
        Self::interpolate_stops(&colors, t)
    }

    /// Plasma color map (simplified approximation).
    fn plasma(t: f32) -> [f32; 3] {
        let colors: [(f32, [f32; 3]); 5] = [
            (0.0, [0.050, 0.030, 0.528]),
            (0.25, [0.494, 0.012, 0.658]),
            (0.5, [0.798, 0.280, 0.470]),
            (0.75, [0.973, 0.585, 0.253]),
            (1.0, [0.940, 0.975, 0.131]),
        ];
        Self::interpolate_stops(&colors, t)
    }

    /// Coolwarm diverging color map.
    fn coolwarm(t: f32) -> [f32; 3] {
        let colors: [(f32, [f32; 3]); 3] = [
            (0.0, [0.230, 0.299, 0.754]),
            (0.5, [0.865, 0.865, 0.865]),
            (1.0, [0.706, 0.016, 0.150]),
        ];
        Self::interpolate_stops(&colors, t)
    }

    /// Simple blue-white water color map.
    fn water(t: f32) -> [f32; 3] {
        let colors: [(f32, [f32; 3]); 3] = [
            (0.0, [0.0, 0.1, 0.4]),
            (0.5, [0.1, 0.4, 0.8]),
            (1.0, [0.7, 0.9, 1.0]),
        ];
        Self::interpolate_stops(&colors, t)
    }

    /// Linear interpolation between color stops.
    fn interpolate_stops(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
        if t <= stops[0].0 {
            return stops[0].1;
        }
        if t >= stops[stops.len() - 1].0 {
            return stops[stops.len() - 1].1;
        }

        for i in 0..stops.len() - 1 {
            let (t0, c0) = stops[i];
            let (t1, c1) = stops[i + 1];
            if t >= t0 && t <= t1 {
                let f = (t - t0) / (t1 - t0);
                return [
                    c0[0] + f * (c1[0] - c0[0]),
                    c0[1] + f * (c1[1] - c0[1]),
                    c0[2] + f * (c1[2] - c0[2]),
                ];
            }
        }

        stops[stops.len() - 1].1
    }
}
